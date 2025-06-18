import json
import time
import csv
import re
from stackapi import StackAPI

def extract_proper_commands(text):
    """Extract actual command-line commands from answer text"""
    if not text:
        return ""
    
    commands = []
    
    # Remove HTML tags first
    clean = re.compile('<.*?>')
    text = re.sub(clean, '', text)
    
    # Replace HTML entities
    text = text.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')
    text = text.replace('&quot;', '"').replace('&#39;', "'").replace('&nbsp;', ' ')
    
    # Look for code patterns (commands within backticks or after $ or #)
    code_patterns = [
        r'`([^`\n]+)`',  # Single backticks
        r'^\s*\$\s+([^\n]+)',  # Lines starting with $
        r'^\s*#\s+([^\n]+)',   # Lines starting with #
        r'^\s*(git\s+[^\n]+)',  # Git commands
        r'^\s*(bash\s+[^\n]+)',  # Bash commands
        r'^\s*(grep\s+[^\n]+)',  # Grep commands
        r'^\s*(tar\s+[^\n]+)',   # Tar commands
        r'^\s*(gzip\s+[^\n]+)',  # Gzip commands
        r'^\s*(python\s+[^\n]+)', # Python commands
        r'^\s*(pip\s+[^\n]+)',   # Pip commands
        r'^\s*(source\s+[^\n]+)', # Source commands
        r'^\s*(\w+\s+-[a-zA-Z][^\n]*)', # Commands with flags
    ]
    
    for pattern in code_patterns:
        matches = re.findall(pattern, text, re.MULTILINE | re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                match = match[0] if match else ""
            
            cmd = match.strip()
            
            # Filter valid commands
            if (5 <= len(cmd) <= 120 and  # Reasonable length
                not cmd.startswith('http') and  # Not URLs
                not cmd.startswith('//') and    # Not comments
                not any(x in cmd.lower() for x in ['example', 'your_', 'path/to', '<br>', 'etc.'])):
                
                # Clean up common prefixes
                cmd = re.sub(r'^\$\s*', '', cmd)  # Remove $ prefix
                cmd = re.sub(r'^#\s*', '', cmd)   # Remove # prefix
                
                commands.append(cmd)
    
    return ' | '.join(commands[:2]) if commands else ""

def generate_comprehensive_qa_pairs():
    """Generate comprehensive Q&A pairs with proper commands"""
    
    # Initialize Stack Overflow API
    api_key = 'rl_PeHiEyssMzH8eUUh479ceTMGp'
    site = StackAPI('stackoverflow', key=api_key)
    site.page_size = 100
    site.max_pages = 2
    
    # Extended tags for better coverage
    command_tags = {
        'git': ['git', 'git-branch', 'git-merge', 'git-rebase'],
        'bash': ['bash', 'shell', 'command-line'],
        'archive': ['tar', 'gzip', 'zip'],
        'search': ['grep', 'find', 'awk', 'sed'],
        'python-env': ['python-venv', 'virtualenv', 'pip']
    }
    
    qa_pairs = []
    target_per_category = 40  # 40 per category = 200 total
    
    print("Generating comprehensive Q&A pairs...")
    
    for category, tags in command_tags.items():
        print(f"\n=== Processing {category.upper()} category ===")
        category_pairs = []
        
        for tag in tags:
            if len(category_pairs) >= target_per_category:
                break
                
            print(f"Fetching questions for tag: {tag}")
            
            try:
                # Get high-quality questions
                questions_data = site.fetch(
                    'questions',
                    tagged=tag,
                    sort='votes',
                    order='desc',
                    filter='withbody',
                    min_score=10  # Only questions with good scores
                )
                
                questions = questions_data.get('items', [])
                print(f"  Found {len(questions)} questions")
                
                for question in questions:
                    if len(category_pairs) >= target_per_category:
                        break
                        
                    try:
                        question_id = question.get('question_id')
                        title = question.get('title', '').strip()
                        
                        if not title or len(title) < 10:
                            continue
                        
                        print(f"  Processing: {title[:60]}...")
                        
                        # Get answers
                        answers_data = site.fetch(
                            'questions/{ids}/answers',
                            ids=[question_id],
                            sort='votes',
                            order='desc',
                            filter='withbody',
                            min_score=5  # Only good answers
                        )
                        
                        answers = answers_data.get('items', [])
                        
                        if answers:
                            # Find best answer (accepted or highest voted)
                            best_answer = None
                            for answer in answers:
                                if answer.get('is_accepted', False):
                                    best_answer = answer
                                    break
                            
                            if not best_answer:
                                best_answer = answers[0]
                            
                            answer_body = best_answer.get('body', '')
                            
                            # Extract commands
                            commands = extract_proper_commands(answer_body)
                            
                            if commands and len(commands) > 5:
                                qa_pair = {
                                    'question': title,
                                    'command': commands,
                                    'category': category,
                                    'tag': tag,
                                    'votes': question.get('score', 0),
                                    'answer_votes': best_answer.get('score', 0)
                                }
                                
                                category_pairs.append(qa_pair)
                                print(f"    ✓ Added: {commands[:50]}...")
                            else:
                                print(f"    ✗ No valid commands found")
                        
                        time.sleep(0.1)  # Rate limiting
                        
                    except Exception as e:
                        print(f"    Error processing question: {str(e)}")
                        continue
                
                time.sleep(0.5)  # Between tags
                
            except Exception as e:
                print(f"  Error fetching {tag}: {str(e)}")
                continue
        
        qa_pairs.extend(category_pairs)
        print(f"Category {category}: {len(category_pairs)} Q&A pairs")
    
    print(f"\n=== SUMMARY ===")
    print(f"Total Q&A pairs generated: {len(qa_pairs)}")
    
    # Save to CSV
    if qa_pairs:
        filename = 'final_command_line_qa_200.csv'
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['question', 'command', 'category', 'tag', 'votes', 'answer_votes']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for qa in qa_pairs:
                writer.writerow(qa)
        
        print(f"✓ Saved to '{filename}'")
        
        # Show statistics
        by_category = {}
        for qa in qa_pairs:
            cat = qa['category']
            by_category[cat] = by_category.get(cat, 0) + 1
        
        print("\nBreakdown by category:")
        for cat, count in by_category.items():
            print(f"  {cat}: {count} pairs")
        
        # Show top examples
        print(f"\nTop 10 Q&A pairs by votes:")
        sorted_qa = sorted(qa_pairs, key=lambda x: x['votes'], reverse=True)
        for i, qa in enumerate(sorted_qa[:10]):
            print(f"\n{i+1}. Q: {qa['question']}")
            print(f"   Command: {qa['command']}")
            print(f"   Category: {qa['category']} | Votes: {qa['votes']}")
    
    return qa_pairs

# Install required library first
import subprocess
import sys

try:
    import stackapi
    print("StackAPI is already installed")
except ImportError:
    print("Installing StackAPI...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "stackapi"])
    import stackapi
    print("StackAPI installed successfully")

# Generate the final dataset
final_qa_data = generate_comprehensive_qa_pairs()
