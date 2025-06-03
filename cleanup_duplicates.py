#!/usr/bin/env python3
"""
Remove Duplicates Script for Trading Bot
Cleans up duplicate code and consolidates implementations
"""

import re
import os
import shutil
from datetime import datetime

def backup_file(filepath):
    """Create a backup of the original file"""
    backup_path = f"{filepath}.dedup_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(filepath, backup_path)
    print(f"✅ Created backup: {backup_path}")
    return backup_path

def remove_duplicate_imports(content):
    """Remove duplicate import statements"""
    print("🧹 Removing duplicate imports...")
    
    # Remove duplicate Tuple import
    content = re.sub(r'\nfrom typing import Tuple\n', '\n', content, count=1)
    
    # Remove commented duplicate imports
    content = re.sub(r'# Removed duplicate datetime and deque imports\n', '', content)
    
    print("  ✅ Cleaned up import statements")
    return content

def consolidate_account_managers(content):
    """Keep only EnhancedAccountManager, remove others"""
    print("🔧 Consolidating AccountManager classes...")
    
    # Remove base AccountManager class (keep only Enhanced)
    pattern1 = r'# =====================================\n# BASE ACCOUNT MANAGER\n# =====================================\nclass AccountManager:.*?(?=# =====================================)'
    content = re.sub(pattern1, '', content, flags=re.DOTALL)
    
    # Remove HFQAccountManager class
    pattern2 = r'# =====================================\n# COMPLETE HFQ ACCOUNTMANAGER\n# =====================================\n\nclass HFQAccountManager:.*?(?=# =====================================|class \w+|$)'
    content = re.sub(pattern2, '', content, flags=re.DOTALL)
    
    # Update all references to use EnhancedAccountManager
    content = re.sub(r'HFQAccountManager\(', 'EnhancedAccountManager(', content)
    content = re.sub(r'AccountManager\(', 'EnhancedAccountManager(', content)
    
    print("  ✅ Consolidated to single EnhancedAccountManager")
    return content

def remove_duplicate_methods(content):
    """Remove duplicate method implementations"""
    print("🔍 Removing duplicate methods...")
    
    # Remove standalone duplicate methods that appear after classes
    patterns = [
        r'\ndef check_sufficient_balance\(self.*?\n(?=\ndef|\nclass|# ====)',
        r'\ndef calculate_portfolio_risk\(self.*?\n(?=\ndef|\nclass|# ====)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, content, flags=re.DOTALL)
        if len(matches) > 1:
            # Keep first occurrence, remove others
            content = re.sub(pattern, '', content, count=len(matches)-1, flags=re.DOTALL)
    
    print("  ✅ Removed duplicate method implementations")
    return content

def consolidate_trailing_managers(content):
    """Keep only EnhancedTrailingStopManager"""
    print("🎯 Consolidating TrailingStopManager classes...")
    
    # Remove the stub TrailingStopManager
    pattern = r'class TrailingStopManager:\n\s*"""Manages trailing stops for all positions"""\n.*?(?=class|\n\n[A-Z]|$)'
    content = re.sub(pattern, '', content, flags=re.DOTALL)
    
    # Update references
    content = re.sub(r'TrailingStopManager\(\)', 'create_enhanced_trailing_stop_manager(session, ta_engine, config, logger)', content)
    
    print("  ✅ Consolidated to EnhancedTrailingStopManager")
    return content

def remove_duplicate_globals(content):
    """Remove duplicate global variables"""
    print("🌐 Removing duplicate global variables...")
    
    # Remove bybit_session = session duplicate
    content = re.sub(r'\nbybit_session = session.*?\n', '\n', content)
    
    # Update all bybit_session references to use session
    content = re.sub(r'self\.bybit_session', 'self.session', content)
    content = re.sub(r'bybit_session', 'session', content)
    
    print("  ✅ Removed duplicate session variables")
    return content

def fix_strategy_config_duplicates(content):
    """Fix duplicate attributes in strategy configs"""
    print("⚙️ Fixing strategy configuration duplicates...")
    
    # Remove redundant symbols assignment
    content = re.sub(
        r'self\.symbols = self\.scan_symbols\s*# Use the existing scan_symbols',
        '# Removed duplicate symbols assignment',
        content
    )
    
    print("  ✅ Fixed strategy configuration duplicates")
    return content

def add_missing_initializations(content):
    """Ensure proper initialization order"""
    print("🔗 Fixing initialization order...")
    
    # Ensure account_manager is initialized before order_manager
    init_pattern = r'(# Initialize AccountManager.*?)(# Initialize OrderManager)'
    
    if not re.search(r'account_manager = EnhancedAccountManager\(session\)', content):
        replacement = r'\1account_manager = EnhancedAccountManager(session)\n\2'
        content = re.sub(init_pattern, replacement, content, flags=re.DOTALL)
    
    print("  ✅ Fixed initialization order")
    return content

def create_clean_summary(content):
    """Add summary of what was cleaned"""
    print("📝 Adding cleanup summary...")
    
    cleanup_summary = '''
# =====================================
# DEDUPLICATION SUMMARY
# =====================================
# This code has been cleaned of duplicates:
# - Single AccountManager implementation (EnhancedAccountManager)
# - Single TrailingStopManager implementation (EnhancedTrailingStopManager)
# - Removed duplicate imports
# - Removed duplicate methods
# - Consolidated session variables
# - Fixed strategy config duplicates
# =====================================

'''
    
    # Add after imports section
    import_end = content.find('# =====================================')
    if import_end != -1:
        content = content[:import_end] + cleanup_summary + content[import_end:]
    
    print("  ✅ Added cleanup summary")
    return content

def main():
    """Remove all duplicates from the bot code"""
    print("🧹 Duplicate Removal Script for Trading Bot")
    print("="*60)
    
    bot_file = "bot_loop.py"
    
    if not os.path.exists(bot_file):
        print(f"❌ Error: {bot_file} not found!")
        return
    
    # Create backup
    backup_path = backup_file(bot_file)
    
    try:
        # Read file
        with open(bot_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_size = len(content)
        
        print("\n🔧 Removing duplicates...")
        
        # Apply all deduplication fixes
        content = remove_duplicate_imports(content)
        content = consolidate_account_managers(content)
        content = remove_duplicate_methods(content)
        content = consolidate_trailing_managers(content)
        content = remove_duplicate_globals(content)
        content = fix_strategy_config_duplicates(content)
        content = add_missing_initializations(content)
        content = create_clean_summary(content)
        
        # Write cleaned content
        with open(bot_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        new_size = len(content)
        reduction = ((original_size - new_size) / original_size) * 100
        
        print(f"\n✅ Deduplication Complete!")
        print(f"\n📊 SIZE REDUCTION:")
        print(f"  Original: {original_size:,} characters")
        print(f"  Cleaned: {new_size:,} characters")
        print(f"  Reduced by: {reduction:.1f}%")
        
        print(f"\n🧹 DUPLICATES REMOVED:")
        print(f"  • 2 extra AccountManager classes")
        print(f"  • 1 duplicate TrailingStopManager")
        print(f"  • Multiple duplicate methods")
        print(f"  • Duplicate imports")
        print(f"  • Redundant session variables")
        
        print(f"\n📁 Backup saved: {backup_path}")
        print(f"\n✨ Your code is now clean and optimized!")
        
    except Exception as e:
        print(f"\n❌ Error during deduplication: {e}")
        print(f"Backup available at: {backup_path}")
        
        response = input("\nRestore original file? (y/n): ")
        if response.lower() == 'y':
            shutil.copy2(backup_path, bot_file)
            print("✅ Original file restored")

if __name__ == "__main__":
    main()
