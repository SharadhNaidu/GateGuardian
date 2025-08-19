#!/usr/bin/env python3
import os
import re

def remove_comments_from_file(file_path):
    """Remove all comments from a Python file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    cleaned_lines = []
    for line in lines:
        # Remove inline comments but preserve strings that contain #
        in_string = False
        quote_char = None
        result = ""
        i = 0
        
        while i < len(line):
            char = line[i]
            
            if char in ['"', "'"] and (i == 0 or line[i-1] != '\\'):
                if not in_string:
                    in_string = True
                    quote_char = char
                elif char == quote_char:
                    in_string = False
                    quote_char = None
                result += char
            elif char == '#' and not in_string:
                break
            else:
                result += char
            i += 1
        
        # Remove trailing whitespace
        result = result.rstrip()
        
        # Keep non-empty lines and lines that aren't just comments
        if result.strip():
            cleaned_lines.append(result + '\n')
        elif not line.strip().startswith('#'):
            cleaned_lines.append('\n')
    
    # Write back to file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(cleaned_lines)
    
    print(f"✅ Cleaned: {file_path}")

def main():
    gate_guardian_dir = r"c:\Users\iamsh\Desktop\GateGuardian"
    
    # List of Python files to clean
    python_files = [
        "add_face.py",
        "unlock_gate.py", 
        "config.py",
        "test_system.py",
        "camera_test.py",
        "quick_test.py"
    ]
    
    print("🧹 Removing comments from GateGuardian Python files...")
    
    for filename in python_files:
        file_path = os.path.join(gate_guardian_dir, filename)
        if os.path.exists(file_path):
            remove_comments_from_file(file_path)
        else:
            print(f"⚠️  File not found: {filename}")
    
    print("✨ All comments removed successfully!")

if __name__ == "__main__":
    main()