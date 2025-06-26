import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def send_test_email(receiver_email: str):
    """
    Sends a test email using SMTP settings from the .env file.
    """
    load_dotenv()

    # --- Email Configuration ---
    smtp_host = os.environ.get('SMTP_HOST')
    smtp_port = int(os.environ.get('SMTP_PORT', 587))
    smtp_user = os.environ.get('SMTP_USER')
    smtp_password = os.environ.get('SMTP_PASSWORD')
    sender_email = os.environ.get('SENDER_EMAIL', smtp_user)

    if not all([smtp_host, smtp_port, smtp_user, smtp_password, sender_email]):
        logging.error("SMTP settings are not fully configured. Please check your .env file.")
        logging.error(f"SMTP_HOST: {smtp_host}")
        logging.error(f"SMTP_PORT: {smtp_port}")
        logging.error(f"SMTP_USER: {smtp_user}")
        logging.error(f"SENDER_EMAIL: {sender_email}")
        logging.error(f"SMTP_PASSWORD: {'********' if smtp_password else None}")
        return

    logging.info(f"Attempting to send email from {sender_email} to {receiver_email} via {smtp_host}:{smtp_port}")

    # --- Create Email Message ---
    message = MIMEMultipart("alternative")
    message["Subject"] = "SMTP Test Email from Dxee Chat"
    message["From"] = sender_email
    message["To"] = receiver_email

    text = "This is a test email to verify your SMTP configuration."
    html = f"""
    <html>
        <body>
            <h2>SMTP Configuration Test</h2>
            <p>This is a test email from the Dxee Chat application to confirm that your SMTP settings are working correctly.</p>
            <p>If you have received this, the test was successful!</p>
            <hr>
            <p><small>This is an automated message.</small></p>
        </body>
    </html>
    """

    part1 = MIMEText(text, "plain")
    part2 = MIMEText(html, "html")
    message.attach(part1)
    message.attach(part2)

    # --- Send Email ---
    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.set_debuglevel(1)  # Enable debug output
            server.starttls()  # Secure the connection
            server.login(smtp_user, smtp_password)
            server.sendmail(sender_email, receiver_email, message.as_string())
            logging.info("✅ Email sent successfully!")
    except smtplib.SMTPAuthenticationError as e:
        logging.error(f"❌ SMTP Authentication Error: {e.smtp_code} - {e.smtp_error.decode()}")
        logging.error("Check your SMTP_USER and SMTP_PASSWORD.")
    except smtplib.SMTPServerDisconnected as e:
        logging.error(f"❌ SMTP Server Disconnected: {e}")
        logging.error("The server unexpectedly disconnected. Check server status or network.")
    except smtplib.SMTPConnectError as e:
        logging.error(f"❌ SMTP Connection Error: {e}")
        logging.error("Failed to connect to the server. Check SMTP_HOST and SMTP_PORT.")
    except Exception as e:
        logging.error(f"❌ An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Get receiver email from command line argument or prompt the user
    import sys
    if len(sys.argv) > 1:
        receiver = sys.argv[1]
    else:
        receiver = input("Enter the recipient's email address: ")
    
    if receiver:
        send_test_email(receiver)
    else:
        logging.warning("No recipient email address provided. Exiting.") 
