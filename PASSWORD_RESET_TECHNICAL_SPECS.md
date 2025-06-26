# Password Reset Feature - Detailed Technical Specifications

## Overview
Implementation of a secure password reset functionality for email-authenticated users in the DeepSchina application.

## Database Schema Changes

### User Document Extensions
```javascript
{
  // Existing fields...
  "reset_token": "abc123def456...", // 32-char secure random string
  "reset_token_expires": ISODate("2025-06-25T15:30:00Z"), // 30 minutes from creation
  "reset_attempts": 2, // Track reset attempts for rate limiting
  "last_reset_request": ISODate("2025-06-25T15:00:00Z") // Last reset request timestamp
}
```

## Backend API Endpoints

### 1. Forgot Password Endpoint
**Route:** `POST /auth/forgot-password`

**Request Body:**
```json
{
  "email": "user@example.com"
}
```

**Response (Success):**
```json
{
  "success": true,
  "message": "If an account with this email exists, you will receive a password reset link."
}
```

**Response (Rate Limited):**
```json
{
  "error": "Too many reset requests. Please wait before trying again.",
  "retry_after": 3600
}
```

### 2. Reset Password Validation Endpoint
**Route:** `GET /auth/reset-password?token=<reset_token>`

**Response (Valid Token):**
```json
{
  "success": true,
  "email_masked": "j***doe@exa***.com",
  "token_valid": true
}
```

**Response (Invalid/Expired Token):**
```json
{
  "error": "Invalid or expired reset token",
  "token_valid": false
}
```

### 3. Reset Password Submission Endpoint
**Route:** `POST /auth/reset-password`

**Request Body:**
```json
{
  "token": "abc123def456...",
  "new_password": "newSecurePassword123"
}
```

**Response (Success):**
```json
{
  "success": true,
  "message": "Password has been reset successfully."
}
```

## Security Specifications

### Token Generation
```python
import secrets
import hashlib
from datetime import datetime, timedelta

def generate_reset_token():
    """Generate cryptographically secure 32-character token"""
    return secrets.token_urlsafe(24)[:32]

def hash_token(token):
    """Hash token for database storage"""
    return hashlib.sha256(token.encode()).hexdigest()
```

### Rate Limiting Rules
- **Per Email:** Maximum 3 reset requests per hour
- **Per IP:** Maximum 10 reset requests per hour
- **Global:** Maximum 100 reset requests per minute

### Token Security
- **Length:** 32 characters (192 bits of entropy)
- **Expiration:** 30 minutes from generation
- **Storage:** Hashed in database using SHA-256
- **Transmission:** HTTPS only, included in URL parameters

## Email Template Specifications

### HTML Email Template
```html
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Password Reset - DeepSchina</title>
</head>
<body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
    <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
        <h2 style="color: #ff9776;">Password Reset Request</h2>
        <p>Hello,</p>
        <p>You requested a password reset for your DeepSchina account. Click the button below to reset your password:</p>
        
        <div style="text-align: center; margin: 30px 0;">
            <a href="{{reset_url}}" 
               style="background-color: #ff9776; color: white; padding: 12px 30px; 
                      text-decoration: none; border-radius: 5px; display: inline-block;">
                Reset Password
            </a>
        </div>
        
        <p>Or copy and paste this link into your browser:</p>
        <p style="word-break: break-all; background-color: #f5f5f5; padding: 10px; border-radius: 3px;">
            {{reset_url}}
        </p>
        
        <p><strong>This link will expire in 30 minutes.</strong></p>
        
        <p>If you didn't request this password reset, please ignore this email. Your password will remain unchanged.</p>
        
        <hr style="border: none; border-top: 1px solid #eee; margin: 30px 0;">
        <p style="font-size: 12px; color: #666;">
            This is an automated message from DeepSchina. Please do not reply to this email.
        </p>
    </div>
</body>
</html>
```

### Plain Text Email Template
```text
Password Reset Request - DeepSchina

Hello,

You requested a password reset for your DeepSchina account. 

Please visit the following link to reset your password:
{{reset_url}}

This link will expire in 30 minutes.

If you didn't request this password reset, please ignore this email. Your password will remain unchanged.

---
This is an automated message from DeepSchina. Please do not reply to this email.
```

## Frontend UI Specifications

### 1. Login Form Modification
```html
<!-- Add to existing login form -->
<div class="forgot-password-link">
    <a href="#" id="forgot-password-link" style="color: #ff9776; font-size: 14px;">
        Forgot Password?
    </a>
</div>
```

### 2. Forgot Password Modal
```html
<div id="forgot-password-modal" class="modal hidden">
    <div class="modal-content">
        <div class="modal-header">
            <h3>Reset Password</h3>
            <button class="close-modal">&times;</button>
        </div>
        <div class="modal-body">
            <form id="forgot-password-form">
                <div class="form-group">
                    <label for="reset-email">Email Address</label>
                    <input type="email" id="reset-email" name="email" required 
                           placeholder="Enter your email address">
                </div>
                <div class="form-actions">
                    <button type="submit" class="btn-primary">Send Reset Link</button>
                    <button type="button" class="btn-secondary" id="back-to-login">
                        Back to Login
                    </button>
                </div>
            </form>
            <div id="reset-success-message" class="success-message hidden">
                <p>If an account with this email exists, you will receive a password reset link shortly.</p>
            </div>
        </div>
    </div>
</div>
```

### 3. Reset Password Page Template
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reset Password - DeepSchina</title>
    <!-- Include existing styles -->
</head>
<body>
    <div class="reset-password-container">
        <div class="reset-form-wrapper">
            <h2>Set New Password</h2>
            <p class="reset-email">Resetting password for: <strong>{{email_masked}}</strong></p>
            
            <form id="reset-password-form">
                <input type="hidden" name="token" value="{{reset_token}}">
                
                <div class="form-group">
                    <label for="new-password">New Password</label>
                    <input type="password" id="new-password" name="new_password" 
                           required minlength="8" placeholder="Enter new password">
                    <div class="password-requirements">
                        <small>Password must be at least 8 characters long</small>
                    </div>
                </div>
                
                <div class="form-group">
                    <label for="confirm-password">Confirm Password</label>
                    <input type="password" id="confirm-password" name="confirm_password" 
                           required placeholder="Confirm new password">
                </div>
                
                <button type="submit" class="btn-primary">Reset Password</button>
            </form>
            
            <div class="back-to-login">
                <a href="/">Back to Login</a>
            </div>
        </div>
    </div>
</body>
</html>
```

## JavaScript Implementation

### 1. Forgot Password Handler
```javascript
// Add to existing auth.js or main JavaScript
function initForgotPassword() {
    const forgotLink = document.getElementById('forgot-password-link');
    const modal = document.getElementById('forgot-password-modal');
    const form = document.getElementById('forgot-password-form');
    const emailInput = document.getElementById('reset-email');
    
    forgotLink.addEventListener('click', (e) => {
        e.preventDefault();
        // Pre-populate email from login form
        const loginEmail = document.getElementById('login-email')?.value;
        if (loginEmail) {
            emailInput.value = loginEmail;
        }
        showModal(modal);
    });
    
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        await handleForgotPassword(form);
    });
}

async function handleForgotPassword(form) {
    const formData = new FormData(form);
    const email = formData.get('email');
    
    try {
        const response = await fetch('/auth/forgot-password', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ email })
        });
        
        const result = await response.json();
        
        if (response.ok) {
            showSuccessMessage();
            setTimeout(() => {
                hideModal();
            }, 3000);
        } else {
            showError(result.error);
        }
    } catch (error) {
        showError('Network error. Please try again.');
    }
}
```

### 2. Reset Password Page Handler
```javascript
function initResetPassword() {
    const form = document.getElementById('reset-password-form');
    const newPassword = document.getElementById('new-password');
    const confirmPassword = document.getElementById('confirm-password');
    
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        if (newPassword.value !== confirmPassword.value) {
            showError('Passwords do not match');
            return;
        }
        
        await handleResetPassword(form);
    });
    
    // Real-time password confirmation validation
    confirmPassword.addEventListener('input', () => {
        if (confirmPassword.value && newPassword.value !== confirmPassword.value) {
            confirmPassword.setCustomValidity('Passwords do not match');
        } else {
            confirmPassword.setCustomValidity('');
        }
    });
}

async function handleResetPassword(form) {
    const formData = new FormData(form);
    const data = {
        token: formData.get('token'),
        new_password: formData.get('new_password')
    };
    
    try {
        const response = await fetch('/auth/reset-password', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        
        const result = await response.json();
        
        if (response.ok) {
            showSuccess('Password reset successfully! Redirecting to login...');
            setTimeout(() => {
                window.location.href = '/';
            }, 2000);
        } else {
            showError(result.error);
        }
    } catch (error) {
        showError('Network error. Please try again.');
    }
}
```

## Error Handling Specifications

### Backend Error Responses
```python
# Standard error response format
{
    "error": "Human-readable error message",
    "error_code": "SPECIFIC_ERROR_CODE",
    "details": {
        "field": "specific_field_error"  # Optional
    }
}

# Error codes:
# - INVALID_EMAIL: Email format is invalid
# - USER_NOT_FOUND: No user with this email (don't reveal this)
# - RATE_LIMITED: Too many requests
# - TOKEN_EXPIRED: Reset token has expired
# - TOKEN_INVALID: Reset token is invalid
# - PASSWORD_TOO_WEAK: Password doesn't meet requirements
# - SERVER_ERROR: Internal server error
```

### Frontend Error Display
```javascript
function showError(message, field = null) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.textContent = message;
    
    if (field) {
        const fieldElement = document.getElementById(field);
        fieldElement.parentNode.appendChild(errorDiv);
        fieldElement.classList.add('error');
    } else {
        // Show general error
        const container = document.querySelector('.form-container');
        container.insertBefore(errorDiv, container.firstChild);
    }
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        errorDiv.remove();
        if (field) {
            document.getElementById(field).classList.remove('error');
        }
    }, 5000);
}
```

## CSS Styling Specifications

### Modal Styles
```css
.modal {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(0, 0, 0, 0.7);
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 1000;
}

.modal-content {
    background-color: #333;
    padding: 30px;
    border-radius: 8px;
    width: 90%;
    max-width: 400px;
    color: #e0e0e0;
}

.modal-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
    border-bottom: 1px solid #444;
    padding-bottom: 10px;
}

.close-modal {
    background: none;
    border: none;
    color: #999;
    font-size: 24px;
    cursor: pointer;
}

.close-modal:hover {
    color: #fff;
}
```

### Form Styles
```css
.form-group {
    margin-bottom: 20px;
}

.form-group label {
    display: block;
    margin-bottom: 5px;
    color: #e0e0e0;
    font-weight: 500;
}

.form-group input {
    width: 100%;
    padding: 12px;
    border: 1px solid #444;
    border-radius: 4px;
    background-color: #222;
    color: #e0e0e0;
    font-size: 16px;
}

.form-group input:focus {
    outline: none;
    border-color: #ff9776;
    box-shadow: 0 0 0 2px rgba(255, 151, 118, 0.2);
}

.form-group input.error {
    border-color: #ff6b6b;
}

.btn-primary {
    background-color: #ff9776;
    color: white;
    padding: 12px 24px;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 16px;
    width: 100%;
    margin-bottom: 10px;
}

.btn-primary:hover {
    background-color: #ff8660;
}

.btn-secondary {
    background-color: transparent;
    color: #999;
    padding: 12px 24px;
    border: 1px solid #444;
    border-radius: 4px;
    cursor: pointer;
    font-size: 16px;
    width: 100%;
}

.btn-secondary:hover {
    color: #fff;
    border-color: #666;
}
```

## Testing Specifications

### Unit Tests Required
1. **Token Generation Tests**
   - Verify token uniqueness
   - Verify token length and format
   - Verify token expiration logic

2. **Email Validation Tests**
   - Valid email formats
   - Invalid email formats
   - Email existence checks

3. **Rate Limiting Tests**
   - Per-email rate limiting
   - Per-IP rate limiting
   - Global rate limiting

4. **Password Reset Flow Tests**
   - Valid token reset
   - Expired token handling
   - Invalid token handling
   - Password validation

### Integration Tests Required
1. **End-to-End Flow Test**
   - Complete password reset journey
   - Email delivery verification
   - Database state verification

2. **Security Tests**
   - Token brute force protection
   - CSRF protection
   - SQL injection prevention

## Deployment Considerations

### Environment Variables
```bash
# Add to .env file
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
SENDER_EMAIL=noreply@deepschina.com
RESET_TOKEN_EXPIRY_MINUTES=30
RESET_RATE_LIMIT_PER_HOUR=3
```

### Database Indexes
```javascript
// MongoDB indexes to add
db.users.createIndex({ "reset_token": 1 }, { sparse: true })
db.users.createIndex({ "reset_token_expires": 1 }, { expireAfterSeconds: 0 })
db.users.createIndex({ "email_hash": 1, "auth_type": 1 })
```

This comprehensive specification covers all aspects of the password reset implementation, ensuring security, usability, and maintainability.
