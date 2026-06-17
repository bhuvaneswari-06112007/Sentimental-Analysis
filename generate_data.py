import os
import pandas as pd
import random

def generate_synthetic_data():
    categories = ['HR', 'Finance', 'IT Support', 'Marketing']
    
    # Templates for synthetic emails
    templates = {
        'HR': [
            ("Job Application: Software Engineer", "Hello, I am interested in applying for the Software Engineer position. Please find my resume attached."),
            ("Onboarding Documents Required", "Welcome to the team! Please complete and submit your onboarding documents as soon as possible."),
            ("Performance Review Feedback", "Hi, the annual performance review process has started. Please fill out your self-evaluation form by Friday."),
            ("Annual Health Insurance Enrollment", "This is a reminder that the open enrollment period for annual health insurance ends next week. Please review the plans."),
            ("Request for Paid Time Off (PTO)", "Hi team, I would like to request PTO from next Monday to Wednesday for personal reasons. Please approve."),
            ("Interview Schedule - Product Manager", "Dear candidate, we would like to schedule a virtual interview with you for the Product Manager role. Please let us know your availability."),
            ("Policy Updates on Remote Work", "Please read the updated employee handbook regarding the new hybrid/remote work guidelines."),
            ("Referral Bonus Program", "Recommend your friends for open positions and receive a referral bonus. Check the internal job portal for details.")
        ],
        'Finance': [
            ("Invoice payment pending", "Hi, the invoice INV-2026-004 is overdue. Please process the payment of $4,500 immediately to avoid late fees."),
            ("Monthly Budget Review", "Dear team, the budget review meeting is scheduled for tomorrow at 10 AM. Please bring your Q3 financial reports."),
            ("Expense Report Approval Request", "I have submitted my travel expense report for last week's conference. Please review and approve it."),
            ("Q2 Financial Statement Draft", "Attached is the draft Q2 financial statement. Please review the numbers and let me know if there are any discrepancies."),
            ("Tax Filing Documents", "Please provide your tax declarations and relevant receipts to the finance department by the end of this month."),
            ("Payroll processing schedule", "This is a notification that payroll will be processed early this month due to the upcoming holidays. Ensure all hours are logged."),
            ("Audit queries response needed", "The internal audit team has raised queries regarding the travel expenses. Please provide supporting receipts."),
            ("Client Billing details updated", "The billing address and bank account details for client Acme Corp have been updated in our system.")
        ],
        'IT Support': [
            ("System Outage: Server Down", "We are experiencing an unexpected system outage on Server-04. The engineering team is actively investigating the issue."),
            ("Reset Password Request", "Hi IT Support, I have locked myself out of my account. Can you please trigger a password reset for me? Thank you."),
            ("Software Installation Permission", "I need to install Docker Desktop on my local machine for a new project. Could you please grant admin rights or install it?"),
            ("VPN connection issue", "I am unable to connect to the corporate VPN from home. I keep getting a timeout error. Please assist."),
            ("Hardware Upgrade: New Laptop", "My current laptop is running extremely slow and has disk issues. I would like to request a hardware upgrade/replacement."),
            ("Phishing Alert: Suspicious Email", "I received a suspicious email asking for my login credentials. I suspect it is a phishing attempt. Forwarding for analysis."),
            ("Printer setup assistance", "Hi, I cannot connect to the office printer on the 3rd floor. Can you help me map the printer driver?"),
            ("Access Request to Git Repo", "Hi, please grant me read and write access to the main codebase repository on GitHub. My username is dev_user.")
        ],
        'Marketing': [
            ("New Campaign Launch Plan", "Hi all, we are launching the summer marketing campaign next week. Attached is the presentation detailing the social media strategy."),
            ("Newsletter Content Draft", "Here is the draft for the monthly newsletter. Please review the copy and design before we send it to subscribers."),
            ("Social Media Analytics - May", "Our social media engagement grew by 15% in May. Here is the report showing top-performing posts and follower growth."),
            ("Brand Guidelines Document", "Attached are the updated brand guidelines, including new logo variations and color palettes. Please use these for all external assets."),
            ("Press Release: Product Launch", "Please find the draft press release for our new feature launch. We plan to distribute it to major media outlets tomorrow."),
            ("SEO optimization report", "We have completed the keyword research and SEO optimization for the landing pages. Organic traffic is expected to increase."),
            ("Customer feedback survey results", "Here is the summary of the customer feedback survey. Most users are happy, but we need to address some usability concerns."),
            ("Event Sponsorship Proposal", "We have received a proposal to sponsor the upcoming Tech Innovators Conference. Please review the package options.")
        ]
    }

    # Add urgency levels to some emails to help the priority engine
    urgency_signals = [
        " URGENT: Action required immediately.",
        " Please look into this ASAP!",
        " This is a high-priority issue.",
        " Critical deadline today.",
        " Safe to ignore or process later."
    ]

    data = []
    # Generate 200 samples
    for _ in range(200):
        category = random.choice(categories)
        subject, body = random.choice(templates[category])
        
        # Add random urgency flags to simulate real priority distribution
        if random.random() < 0.3:
            signal = random.choice(urgency_signals)
            if random.random() < 0.5:
                subject = f"[URGENT] {subject}"
            body = f"{body}{signal}"
            
        data.append({
            'subject': subject,
            'body': body,
            'category': category
        })
        
    df = pd.DataFrame(data)
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/synthetic_emails.csv', index=False)
    print(f"Generated {len(df)} synthetic emails in 'data/synthetic_emails.csv'")

if __name__ == '__main__':
    generate_synthetic_data()
