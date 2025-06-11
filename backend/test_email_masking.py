#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify the new email masking pattern
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.service.mongodb_service import MongoDBService

def test_email_masking():
    """Test the email masking function with various email formats"""
    mongodb = MongoDBService()
    
    test_emails = [
        "john.doe@example.com",
        "donkun77jp@gmail.com", 
        "a@b.com",
        "test@domain.co.uk",
        "very.long.email.address@subdomain.example.org",
        "short@ex.net",
        "user123@company.com"
    ]
    
    print("? Testing Email Masking Pattern:")
    print("Pattern: first char + *** + last 3 chars @ first 3 chars + *** + domain extension")
    print("=" * 70)
    
    for email in test_emails:
        masked = mongodb._mask_email(email)
        print(f"Original: {email:<35} �� Masked: {masked}")
    
    print("=" * 70)
    print("? Email masking test completed!")

if __name__ == "__main__":
    test_email_masking()
