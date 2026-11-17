"""Writes data/pairs.yaml. All pairs are synthetic and fictional.

Written by hand BEFORE any similarity score was computed (see DECISIONS.md). Texts below are the
source of truth; run `python scripts/make_pairs.py` to regenerate the YAML.
"""

from pathlib import Path

import yaml

OUT = Path(__file__).resolve().parent.parent / "data" / "pairs.yaml"
OUT_ADV = OUT.with_name("adversarial_hits.yaml")

# ---- hit types -------------------------------------------------------------------------------

PARAPHRASE = [
    ("What is the refund window for EU orders?", "How long do I have to return something bought in the EU?"),
    ("How do I change the email address on the account?", "Where can I update the account email?"),
    ("Does the Pro plan include priority support?", "Is priority support part of the Pro plan?"),
    ("How can I export all project data as CSV?", "What is the way to download project data in CSV format?"),
    ("Which payment methods are accepted?", "What ways of paying do you support?"),
    ("How do I invite a teammate to the workspace?", "What steps add a new member to the workspace?"),
    ("Where can I find the API documentation?", "Which page has the API docs?"),
    ("How do I delete a project permanently?", "What is the procedure for permanently removing a project?"),
    ("What browsers does the dashboard support?", "Which web browsers work with the dashboard?"),
    ("Can the invoice be sent to a different address?", "Is it possible to mail the invoice to another address?"),
    ("How do I turn on two-factor authentication?", "What do I need to do to enable 2FA?"),
    ("What is the maximum file size for uploads?", "How large can an uploaded file be?"),
    ("How do I switch to the annual billing cycle?", "What is the process for moving to yearly billing?"),
    ("Is there a mobile app for Android?", "Do you offer an Android application?"),
    ("How do I connect the tool to Slack?", "What are the steps to integrate with Slack?"),
    ("Where do I download the latest invoice?", "How can I get the most recent invoice?"),
    ("What happens when the free trial ends?", "What occurs after the trial period is over?"),
    ("How do I rename a workspace?", "What is the way to change a workspace name?"),
    ("Can I restore a deleted file?", "Is it possible to recover a file that was removed?"),
    ("Does the service support single sign-on?", "Is SSO available?"),
    ("What is the uptime guarantee for the Enterprise plan?", "How reliable is the Enterprise plan in terms of uptime?"),
    ("How do I generate a new API key?", "What is the way to create a fresh API key?"),
    ("Which languages is the interface available in?", "What languages can the UI be displayed in?"),
    ("How do I cancel the subscription?", "What is the process to end a subscription?"),
    ("How do I add a custom domain?", "What steps are needed to use my own domain name?"),
    ("Where are the servers located?", "In which locations is the data hosted?"),
    ("How do I set up a webhook?", "What is the procedure for configuring a webhook?"),
    ("What is the difference between roles and permissions?", "How do roles differ from permissions?"),
    ("Is there a limit on the number of projects?", "How many projects can be created in total?"),
    ("How do I reset the admin password for the workspace?", "What is the way to recover the workspace admin password?"),
    ("Does the platform comply with GDPR?", "Is the platform GDPR compliant?"),
    ("How can I see who changed a document?", "Where do I find the edit history of a document?"),
    ("What is the process for requesting a refund?", "How do I ask for a refund?"),
    ("How do I add a second payment card?", "What are the steps to register another credit card?"),
    ("Can several people edit a document at once?", "Is real-time collaboration on documents supported?"),
    ("Where can I read the privacy policy?", "Which page contains the privacy policy?"),
    ("How do I change the default language?", "What is the way to set a different default language?"),
    ("How do I contact customer support?", "What is the best way to reach the support team?"),
    ("Is the data encrypted at rest?", "Do you encrypt stored data?"),
    ("How do I archive an old project?", "What is the way to put an old project into the archive?"),
    ("What does the Starter plan cost per month?", "How much is the monthly price of the Starter plan?"),
    ("Can I pause my subscription for a while?", "Is it possible to temporarily suspend a subscription?"),
    ("How do I schedule a report to be emailed weekly?", "What is the way to get a report by email every week?"),
    ("Where do I find the audit log?", "Which menu shows the audit log?"),
    ("How do I remove a member from the team?", "What steps take a person off the team?"),
    ("What file formats can be imported?", "Which formats are supported for import?"),
    ("How do I change the notification settings?", "Where are notifications configured?"),
    ("What is a workspace?", "Can you explain what a workspace is?"),
    ("How do I upgrade to the Enterprise plan?", "What is the way to move up to the Enterprise plan?"),
    ("Is there a discount for non-profit organizations?", "Do charities get reduced pricing?"),
    ("How do I use the keyboard shortcuts?", "Which shortcuts can be used with the keyboard?"),
    ("Can the dashboard be shared with someone outside the company?", "Is it possible to give an external person access to the dashboard?"),
    ("How do I clear the cache in the app?", "What is the way to empty the app cache?"),
    ("What is the rate limit for the public API?", "How many requests are allowed per period by the public API?"),
    ("Why can't I log in?", "Why am I unable to log in?"),
    ("How do I add a logo to the invoice?", "What is the way to put a logo on the invoice?"),
    ("Does the tool work offline?", "Can I use the tool without an internet connection?"),
    ("How do I tag items in bulk?", "What is the way to apply tags to many items at once?"),
    ("Is it not possible to merge two projects?", "Can two projects not be merged?"),
    ("Where can I see the status of the service?", "Which page shows the service status?"),
    ("How do I mute a channel?", "What is the way to silence a channel?"),
    ("What should I do if the page will not load?", "How do I fix a page that fails to load?"),
    ("Is there a way to bulk delete old files?", "Can many old files be deleted in one go?"),
    ("How are taxes shown on the invoice?", "Where do the taxes appear on an invoice?"),
    ("What are the system requirements for the desktop app?", "Which computers can run the desktop app?"),
    ("How do I give a colleague read-only access?", "What is the way to let a coworker only view things?"),
    ("Who can see the shared folders?", "Which people have visibility of shared folders?"),
    ("How do I leave a workspace?", "What is the process for exiting a workspace?"),
    ("What is the data retention period for logs?", "How long are logs kept?"),
]

REORDER = [
    ("What is the refund window for EU orders?", "For EU orders, what is the refund window?"),
    ("How do I export project data as CSV?", "As CSV, how do I export project data?"),
    ("Does the Pro plan include priority support?", "Priority support, does the Pro plan include it?"),
    ("How do I connect the tool to Slack?", "To Slack, how do I connect the tool?"),
    ("Where can I download the latest invoice?", "The latest invoice, where can I download it?"),
    ("Which payment methods are accepted for the Starter plan?", "For the Starter plan, which payment methods are accepted?"),
    ("How do I add a custom domain to a workspace?", "To a workspace, how do I add a custom domain?"),
    ("Can I restore a deleted file from the trash?", "From the trash, can I restore a deleted file?"),
    ("How do I generate an API key for the sandbox?", "For the sandbox, how do I generate an API key?"),
    ("What is the maximum file size for uploads in the Pro plan?", "In the Pro plan, what is the maximum file size for uploads?"),
    ("Does the Enterprise plan support single sign-on?", "Single sign-on, does the Enterprise plan support it?"),
    ("How do I change the billing address on an invoice?", "On an invoice, how do I change the billing address?"),
    ("Is there a mobile app for iOS and Android?", "For iOS and Android, is there a mobile app?"),
    ("How do I archive old projects in the dashboard?", "In the dashboard, how do I archive old projects?"),
    ("Where do I find the audit log for admins?", "For admins, where do I find the audit log?"),
    ("How can I invite a teammate by email?", "By email, how can I invite a teammate?"),
    ("What is the uptime guarantee in the service agreement?", "In the service agreement, what is the uptime guarantee?"),
    ("How do I delete a workspace permanently?", "Permanently, how do I delete a workspace?"),
    ("What languages does the interface support on mobile?", "On mobile, what languages does the interface support?"),
    ("How do I set up a webhook for new orders?", "For new orders, how do I set up a webhook?"),
    ("What is the process for requesting a refund by email?", "By email, what is the process for requesting a refund?"),
    ("How do I switch to annual billing in the settings?", "In the settings, how do I switch to annual billing?"),
    ("Can several people edit one document at the same time?", "At the same time, can several people edit one document?"),
    ("How long are logs kept in the Enterprise plan?", "In the Enterprise plan, how long are logs kept?"),
    ("How do I rename a project in the sidebar?", "In the sidebar, how do I rename a project?"),
    ("Which browsers work with the dashboard on Windows?", "On Windows, which browsers work with the dashboard?"),
    ("Where are notifications configured for the team?", "For the team, where are notifications configured?"),
    ("How do I remove a member from the workspace?", "From the workspace, how do I remove a member?"),
    ("How do I turn on two-factor authentication for admins?", "For admins, how do I turn on two-factor authentication?"),
    ("What does the Basic plan include for small teams?", "For small teams, what does the Basic plan include?"),
    ("Where do I read the privacy policy for the website?", "For the website, where do I read the privacy policy?"),
    ("How do I tag items in bulk in the editor?", "In the editor, how do I tag items in bulk?"),
    ("Is the data encrypted at rest in the US region?", "In the US region, is the data encrypted at rest?"),
    ("How do I upgrade the plan from the billing page?", "From the billing page, how do I upgrade the plan?"),
    ("Which file formats can be imported into a project?", "Into a project, which file formats can be imported?"),
    ("What is the rate limit for the public API on free accounts?", "On free accounts, what is the rate limit for the public API?"),
    ("How do I clear the cache in the desktop app?", "In the desktop app, how do I clear the cache?"),
    ("Can I schedule a report to be emailed every Monday?", "Every Monday, can I schedule a report to be emailed?"),
    ("Where can I see the service status page?", "The service status page, where can I see it?"),
    ("How do I add a second payment card in the settings?", "In the settings, how do I add a second payment card?"),
    ("Does the tool work offline on a laptop?", "On a laptop, does the tool work offline?"),
]

TYPO_BASES = [
    "What is the refund window for EU orders?",
    "How do I export project data as CSV?",
    "Does the Pro plan include priority support?",
    "How do I connect the tool to Slack?",
    "Where can I download the latest invoice?",
    "Which payment methods are accepted?",
    "How do I add a custom domain?",
    "Can I restore a deleted file?",
    "How do I generate a new API key?",
    "What is the maximum file size for uploads?",
    "Does the Enterprise plan support single sign-on?",
    "How do I change the billing address?",
    "Is there a mobile app for Android?",
    "How do I archive an old project?",
    "Where do I find the audit log?",
    "How can I invite a teammate to the workspace?",
    "What is the uptime guarantee?",
    "How do I delete a workspace permanently?",
    "Which languages does the interface support?",
    "How do I set up a webhook?",
    "What is the process for requesting a refund?",
    "How do I switch to annual billing?",
    "Can several people edit a document at once?",
    "How long are logs kept?",
    "How do I rename a project?",
    "Which browsers work with the dashboard?",
    "Where are notifications configured?",
    "How do I remove a member from the workspace?",
    "How do I turn on two-factor authentication?",
    "What does the Basic plan include?",
    "Where can I read the privacy policy?",
    "How do I tag items in bulk?",
    "Is the data encrypted at rest?",
    "How do I upgrade the plan?",
    "Which file formats can be imported?",
    "What is the rate limit for the public API?",
    "How do I clear the cache in the app?",
    "Can I schedule a report to be emailed weekly?",
    "Where can I see the service status?",
    "How do I add a second payment card?",
]
# One typo each: (word that is damaged, damaged form). Chosen by hand per base question.
TYPO_EDITS = [
    ("refund", "refnud"), ("export", "exprot"), ("priority", "priorty"), ("connect", "conect"),
    ("download", "downlaod"), ("payment", "paymnet"), ("custom", "cutsom"), ("restore", "restroe"),
    ("generate", "genrate"), ("maximum", "maxmium"), ("single", "sinlge"), ("billing", "bliling"),
    ("mobile", "mobil"), ("archive", "archvie"), ("audit", "auidt"), ("invite", "invtie"),
    ("guarantee", "garantee"), ("permanently", "permanetly"), ("interface", "inteface"),
    ("webhook", "webhok"), ("process", "proces"), ("annual", "anual"), ("several", "severl"),
    ("long", "lnog"), ("rename", "renmae"), ("browsers", "browsres"), ("notifications", "notifcations"),
    ("remove", "remvoe"), ("authentication", "authentification"), ("include", "inlcude"),
    ("privacy", "privcay"), ("items", "itmes"), ("encrypted", "encrpyted"), ("upgrade", "upgarde"),
    ("formats", "fromats"), ("limit", "limt"), ("cache", "cahce"), ("schedule", "schedlue"),
    ("status", "staus"), ("second", "secnod"),
]

CASE_BASES = [
    "What is the refund window for EU orders?",
    "How do I export project data as CSV?",
    "Does the Pro plan include priority support?",
    "How do I connect the tool to Slack?",
    "Where can I download the latest invoice?",
    "Which payment methods are accepted?",
    "How do I add a custom domain?",
    "Can I restore a deleted file?",
    "How do I generate a new API key?",
    "What is the maximum file size for uploads?",
    "Does the Enterprise plan support single sign-on?",
    "How do I change the billing address?",
    "Is there a mobile app for Android?",
    "How do I archive an old project?",
    "Where do I find the audit log?",
    "How can I invite a teammate to the workspace?",
    "What is the uptime guarantee?",
    "How do I delete a workspace permanently?",
    "Which languages does the interface support?",
    "How do I set up a webhook?",
    "What is the process for requesting a refund?",
    "How do I switch to annual billing?",
    "Can several people edit a document at once?",
    "How long are logs kept?",
    "How do I rename a project?",
    "Which browsers work with the dashboard?",
    "Where are notifications configured?",
    "How do I remove a member from the workspace?",
    "How do I turn on two-factor authentication?",
    "What does the Basic plan include?",
]

# ---- miss types ------------------------------------------------------------------------------

ENTITY_SWAP = [
    # entities present in the default gazetteer
    ("What is the refund window for EU orders?", "What is the refund window for US orders?"),
    ("How long does shipping to Germany take?", "How long does shipping to Japan take?"),
    ("Is the Pro plan eligible for a free trial?", "Is the Enterprise plan eligible for a free trial?"),
    ("What is the VAT rate for customers in France?", "What is the VAT rate for customers in India?"),
    ("Does Product A support single sign-on?", "Does Product B support single sign-on?"),
    ("What is the warranty period in the UK?", "What is the warranty period in Canada?"),
    ("Which payment methods work in Australia?", "Which payment methods work in Germany?"),
    ("How much does the Starter plan cost per month?", "How much does the Enterprise plan cost per month?"),
    ("Where is the support office for Europe located?", "Where is the support office for America located?"),
    ("Can Model X be used offline?", "Can Model Y be used offline?"),
    ("What are the data residency rules in the EU?", "What are the data residency rules in the US?"),
    ("Does the Basic plan include API access?", "Does the Pro plan include API access?"),
    ("How do I return an item bought in Japan?", "How do I return an item bought in Canada?"),
    ("What is the delivery fee to the UK?", "What is the delivery fee to Australia?"),
    ("Is Widget A compatible with the mobile app?", "Is Widget B compatible with the mobile app?"),
    ("What taxes apply to orders from India?", "What taxes apply to orders from France?"),
    ("Does the Pro plan include audit logs?", "Does the Starter plan include audit logs?"),
    ("What is the customer support phone number for the US?", "What is the customer support phone number for the UK?"),
    ("How fast is delivery to Canada?", "How fast is delivery to Germany?"),
    ("Is Product B available for annual billing?", "Is Product A available for annual billing?"),
    ("What languages does support offer in Japan?", "What languages does support offer in France?"),
    ("Which currency is used for orders in the EU?", "Which currency is used for orders in the UK?"),
    ("What is the battery life of Model Y?", "What is the battery life of Model X?"),
    ("Can I get an invoice for the Enterprise plan in India?", "Can I get an invoice for the Enterprise plan in Australia?"),
    # entities NOT in the default gazetteer (guards cannot catch these)
    ("How long does shipping to Brazil take?", "How long does shipping to Norway take?"),
    ("What is the return policy for customers in Mexico?", "What is the return policy for customers in Kenya?"),
    ("Do you have a warehouse in Singapore?", "Do you have a warehouse in Chile?"),
    ("Which carriers deliver to Norway?", "Which carriers deliver to Mexico?"),
    ("Is there a local support number in Kenya?", "Is there a local support number in Brazil?"),
    ("What are the import duties for Chile?", "What are the import duties for Singapore?"),
]

NUMBER_CHANGE = [
    ("Can I return an item within 30 days?", "Can I return an item within 60 days?"),
    ("Does version 2 of the API support webhooks?", "Does version 3 of the API support webhooks?"),
    ("Is there a 14-day free trial?", "Is there a 30-day free trial?"),
    ("What is the limit for 100 users?", "What is the limit for 500 users?"),
    ("How do I migrate from v2 to v3?", "How do I migrate from v1 to v2?"),
    ("What does the 10 GB storage tier cost?", "What does the 100 GB storage tier cost?"),
    ("Can I invite up to 5 teammates?", "Can I invite up to 50 teammates?"),
    ("Does the plan include 3 projects?", "Does the plan include 10 projects?"),
    ("How do I upgrade to release 4.1?", "How do I upgrade to release 4.2?"),
    ("Is the guarantee valid for 12 months?", "Is the guarantee valid for 24 months?"),
    ("What happens after 3 failed login attempts?", "What happens after 10 failed login attempts?"),
    ("What is the rate limit of 60 requests per minute?", "What is the rate limit of 600 requests per minute?"),
    ("Can I store files up to 2 GB?", "Can I store files up to 20 GB?"),
    ("How do I renew the 1-year licence?", "How do I renew the 3-year licence?"),
    ("Is support available 5 days a week?", "Is support available 7 days a week?"),
    ("What is the fee for orders under 50 dollars?", "What is the fee for orders under 500 dollars?"),
    ("How do I install Python 3.11?", "How do I install Python 3.12?"),
    ("When does the 7-day return period start?", "When does the 90-day return period start?"),
    ("Can I add 2 payment cards?", "Can I add 4 payment cards?"),
    ("How do I reset the 4-digit PIN?", "How do I reset the 6-digit PIN?"),
    ("Does the Pro plan allow 25 integrations?", "Does the Pro plan allow 250 integrations?"),
    ("What changed in version 1.5 of the app?", "What changed in version 1.6 of the app?"),
    ("Is shipping free over 40 dollars?", "Is shipping free over 80 dollars?"),
    ("How long is the 2 hour support window?", "How long is the 8 hour support window?"),
    ("Can the report cover 6 months?", "Can the report cover 18 months?"),
    ("Does the first 20 percent discount apply to everything?", "Does the first 50 percent discount apply to everything?"),
    ("How do I order 12 licences?", "How do I order 120 licences?"),
    ("What is the price for the 3 seat bundle?", "What is the price for the 30 seat bundle?"),
    ("Is the data kept for 90 days?", "Is the data kept for 365 days?"),
    ("How do I upgrade from iOS 16?", "How do I upgrade from iOS 17?"),
]

NEGATION = [
    ("Can I cancel my plan after the trial starts?", "Can't I cancel my plan after the trial starts?"),
    ("Is the export feature available on the Basic plan?", "Is the export feature not available on the Basic plan?"),
    ("Do I need to verify my email?", "Do I not need to verify my email?"),
    ("Should I enable two-factor authentication?", "Should I never enable two-factor authentication?"),
    ("Can the invoice be edited after it is sent?", "Can the invoice not be edited after it is sent?"),
    ("Is the data shared with third parties?", "Is the data never shared with third parties?"),
    ("Can I use the API with a free account?", "Can I not use the API with a free account?"),
    ("Does the app work with cookies?", "Does the app work without cookies?"),
    ("Is the subscription refundable?", "Is the subscription non-refundable?"),
    ("Are admins able to see private messages?", "Are admins unable to see private messages?"),
    ("Do you store card numbers?", "Do you never store card numbers?"),
    ("Can I log in without a password?", "Can I log in with a password?"),
    ("Is it allowed to share one licence between users?", "Is it not allowed to share one licence between users?"),
    ("Will the price increase next quarter?", "Won't the price increase next quarter?"),
    ("Can I delete a project that has members?", "Can't I delete a project that has members?"),
    ("Is the beta feature enabled by default?", "Is the beta feature not enabled by default?"),
    ("Do notifications go out on weekends?", "Do notifications not go out on weekends?"),
    ("Should I back up files before upgrading?", "Should I not back up files before upgrading?"),
    ("Can viewers comment on documents?", "Can viewers not comment on documents?"),
    ("Does the plan include support?", "Does the plan not include support?"),
    ("Is it possible to merge two accounts?", "Is it impossible to merge two accounts?"),
    ("Can I install the plugin on the free tier?", "Can't I install the plugin on the free tier?"),
    ("Is there a fee for cancelling?", "Is there no fee for cancelling?"),
    ("Does the export include deleted items?", "Does the export not include deleted items?"),
    ("Can exports be scheduled?", "Can exports never be scheduled?"),
    ("Is the service available in our region?", "Is the service not available in our region?"),
    ("Do I have to accept the terms to continue?", "Do I not have to accept the terms to continue?"),
    ("Can external users see the dashboard?", "Can't external users see the dashboard?"),
    ("Are uploads encrypted?", "Are uploads not encrypted?"),
    ("Is sharing with the whole company allowed?", "Is sharing with the whole company never allowed?"),
]

TIME_SCOPE = [
    ("What orders were placed today?", "What orders were placed last year?"),
    ("How many sign-ups did we get this week?", "How many sign-ups did we get last week?"),
    ("What is the service status right now?", "What was the service status yesterday?"),
    ("What is the weather in the forecast for tomorrow?", "What is the weather in the forecast for today?"),
    ("Show the revenue for this month.", "Show the revenue for last month."),
    ("Which features are planned for this year?", "Which features were released last year?"),
    ("What is the current price of the Pro plan?", "What was the price of the Pro plan in 2022?"),
    ("Who is on call today?", "Who was on call yesterday?"),
    ("How many tickets are open today?", "How many tickets were closed last quarter?"),
    ("Is the office open tomorrow?", "Was the office open yesterday?"),
    ("What are the opening hours this week?", "What were the opening hours last week?"),
    ("What is the exchange rate now?", "What was the exchange rate last year?"),
    ("How many users are online currently?", "How many users were online yesterday?"),
    ("What is the latest release?", "What was the first release?"),
    ("What discounts are running this month?", "What discounts ran last month?"),
    ("What meetings are scheduled for today?", "What meetings are scheduled for tomorrow?"),
    ("Did the deployment succeed today?", "Did the deployment succeed last week?"),
    ("How busy is support this morning?", "How busy was support last night?"),
    ("What is the traffic this quarter?", "What was the traffic last quarter?"),
    ("Show the error rate for today.", "Show the error rate for last year."),
    ("What is the deadline for this month?", "What was the deadline for last month?"),
    ("How many orders shipped this year?", "How many orders shipped last year?"),
    ("What is trending right now?", "What was trending in 2020?"),
    ("Who joined the team recently?", "Who joined the team last year?"),
    ("Is the sale on today?", "Was the sale on yesterday?"),
    ("What is the stock level now?", "What was the stock level last month?"),
    ("Which tasks are due tomorrow?", "Which tasks were due yesterday?"),
    ("What is the uptime this week?", "What was the uptime last quarter?"),
    ("How many returns came in this month?", "How many returns came in last year?"),
    ("What is the plan for next year?", "What was the plan for last year?"),
]

PERSONAL = [
    ("Where is my order?", "Where is my order?"),
    ("What is the status of my order?", "What's the status of my order?"),
    ("When will my package arrive?", "When is my package going to arrive?"),
    ("How do I cancel my subscription?", "How can I cancel my subscription?"),
    ("What is my current balance?", "How much is my current balance?"),
    ("Can I change the delivery address on my order?", "Is it possible to change the delivery address on my order?"),
    ("Show my latest invoice.", "Display my most recent invoice."),
    ("Did my payment go through?", "Was my payment successful?"),
    ("Why was my account locked?", "What caused my account to be locked?"),
    ("When does my plan renew?", "What is the renewal date of my plan?"),
    ("Where can I see my order history?", "Which page lists my order history?"),
    ("I ordered a laptop last week, where is it?", "I ordered a laptop last week, what is its location?"),
    ("What did I order last time?", "Which items did I order last time?"),
    ("How many seats are left on my licence?", "How many seats remain on my licence?"),
    ("Is my refund approved?", "Has my refund been approved?"),
    ("What is my username?", "Which username do I have?"),
    ("Which plan am I on?", "What plan do I have?"),
    ("Can you resend my receipt?", "Could you send my receipt again?"),
    ("How much storage have I used?", "What is my storage usage?"),
    ("Has my package shipped?", "Was my package shipped?"),
    ("What is the tracking number for my order?", "Which tracking number belongs to my order?"),
    ("Update my phone number.", "Change my phone number."),
    ("What are my notification settings?", "Which notification settings do I have?"),
    ("When was my last login?", "What time did I last log in?"),
    ("Why was my card declined?", "What is the reason my card was declined?"),
    ("Who is the owner of my workspace?", "Which person owns my workspace?"),
    ("How do I download my data?", "What is the way to download my data?"),
    ("Is my order eligible for a return?", "Can my order be returned?"),
    ("What is my support ticket status?", "Where does my support ticket stand?"),
    ("I want to delete my account, how?", "I want to delete my account, what is the process?"),
]

TOPIC_NEAR = [
    ("How do I reset the password?", "How do I change the password?"),
    ("How do I export data as CSV?", "How do I import data from CSV?"),
    ("How do I add a user to the workspace?", "How do I remove a user from the workspace?"),
    ("What is the refund policy?", "What is the cancellation policy?"),
    ("How do I enable dark mode?", "How do I enable notifications?"),
    ("Where can I see the invoice?", "Where can I see the payment method?"),
    ("How do I create an API key?", "How do I revoke an API key?"),
    ("What is the maximum upload size?", "What is the maximum download size?"),
    ("How do I share a document?", "How do I lock a document?"),
    ("How do I upgrade the plan?", "How do I downgrade the plan?"),
    ("What does the dashboard show?", "What does the report show?"),
    ("How do I create a project?", "How do I duplicate a project?"),
    ("Where do I find the terms of service?", "Where do I find the privacy policy?"),
    ("How do I enable two-factor authentication?", "How do I disable two-factor authentication?"),
    ("What is the uptime guarantee?", "What is the support response time?"),
    ("How do I add a payment card?", "How do I remove a payment card?"),
    ("How do I create a webhook?", "How do I test a webhook?"),
    ("Which formats can be exported?", "Which formats can be previewed?"),
    ("How do I rename a file?", "How do I move a file?"),
    ("What is a workspace?", "What is a project?"),
    ("How do I send an invitation?", "How do I accept an invitation?"),
    ("How do I connect the calendar?", "How do I connect the email inbox?"),
    ("How do I create a backup?", "How do I restore from a backup?"),
    ("What does an admin role do?", "What does a viewer role do?"),
    ("How do I filter the list by date?", "How do I sort the list by date?"),
    ("How do I log in with a passkey?", "How do I log in with a QR code?"),
    ("What is the maximum number of projects?", "What is the minimum number of seats?"),
    ("How do I install the desktop app?", "How do I uninstall the desktop app?"),
    ("How do I create a report?", "How do I delete a report?"),
    ("Where do I change the language?", "Where do I change the time zone?"),
]


# ---- adversarial hits: held-out, written to probe where guards could misfire ---------------------
# All are `hit` pairs (the answer is the same). Written independently of the guard word lists.
ADVERSARIAL = {
    "number_words": [
        ("Can I return an item within thirty days?", "Can I return an item within 30 days?"),
        ("Can I add twenty-five users to the team?", "Can I add 25 users to the team?"),
        ("Is a fifteen minute break included in the session?", "Is a 15 minute break included in the session?"),
        ("Does the licence cover a hundred seats?", "Does the licence cover 100 seats?"),
        ("Is there a seventy percent discount for schools?", "Is there a 70 percent discount for schools?"),
        ("Are backups kept for eighty days?", "Are backups kept for 80 days?"),
        ("Can I attach nineteen files to one ticket?", "Can I attach 19 files to one ticket?"),
        ("What happens after forty-two failed requests?", "What happens after 42 failed requests?"),
        ("Do you ship orders under $50?", "Do you ship orders under 50 dollars?"),
        ("Can the plan cover 1,000 users?", "Can the plan cover 1000 users?"),
    ],
    "version_words": [
        ("Is v3 of the API stable?", "Is version three of the API stable?"),
        ("How do I move to release v2?", "How do I move to release version two?"),
        ("Does API v1 still receive fixes?", "Does API version 1 still receive fixes?"),
        ("What breaks when I upgrade to v4.1?", "What breaks when I upgrade to version 4.1?"),
        ("Is v5 ready for production?", "Is version five ready for production?"),
    ],
    "quarter": [
        ("When are the Q3 results published?", "When are the third quarter results published?"),
        ("Where is the Q1 report?", "Where is the first quarter report?"),
        ("What is in the Q4 plan?", "What is in the fourth quarter plan?"),
        ("Who approves the 2nd quarter budget?", "Who approves the Q2 budget?"),
        ("What is the Q3 target?", "What is the quarter 3 target?"),
    ],
    "time_synonym": [
        ("How many tickets were closed last week?", "How many tickets were closed in the previous week?"),
        ("Show orders from yesterday.", "Show orders from the previous day."),
        ("What is the service status right now?", "What is the service status at the moment?"),
        ("What changes next month?", "What changes in the coming month?"),
        ("How many sign-ups did we have last year?", "How many sign-ups did we have in the previous year?"),
        ("What is the plan this year?", "What is the plan for the current year?"),
        ("Who is on call today?", "Who is on call this day?"),
    ],
    "negation_synonym": [
        ("Why can't I export the report?", "Why am I unable to export the report?"),
        ("Does the old plan no longer include support?", "Does the old plan not include support anymore?"),
        ("Why isn't the invoice showing up?", "Why is the invoice not showing up?"),
        ("Why doesn't the sync run overnight?", "Why does the sync not run overnight?"),
        ("Why cannot the file be shared?", "Why can not the file be shared?"),
        ("Why won't the page load?", "Why will the page not load?"),
        ("Can I sign in without a password?", "Can I sign in with no password?"),
        ("Why does the export never finish?", "Why does the export not ever finish?"),
    ],
    "date_format": [
        ("What changed on 2024-01-05?", "What changed on January 5, 2024?"),
        ("What shipped on 5 Jan 2024?", "What shipped on 2024-01-05?"),
        ("What is due on 03/04/2024?", "What is due on 3/4/2024?"),
        ("What happens on December 25?", "What happens on 25 December?"),
        ("What is open on March 3rd?", "What is open on March 3?"),
        ("When does the contract end on 2025-12-31?", "When does the contract end on 31 December 2025?"),
    ],
    "entity_forms": [
        ("What is the refund window for the EU?", "What is the refund window for the European Union?"),
        ("Do you ship to the US?", "Do you ship to the United States?"),
        ("Is there a UK support line?", "Is there a United Kingdom support line?"),
        ("What taxes apply to European customers?", "What taxes apply to customers in the EU?"),
        ("What do Americans pay for delivery?", "What do US customers pay for delivery?"),
        ("Which perks do Enterprises get?", "Which perks does the Enterprise plan give?"),
        ("What do Pro plans cost?", "What does the Pro plan cost?"),
    ],
    "alnum_codes": [
        ("Can I upload MP3 files?", "Can I upload mp3 files?"),
        ("How do I create an S3 bucket?", "How do I create an s3 bucket?"),
        ("Do you support IPv6?", "Do you support ipv6?"),
    ],
}


def _typo(base: str, word: str, bad: str) -> str:
    assert word in base, (word, base)
    return base.replace(word, bad, 1)


def build() -> list[dict]:
    pairs: list[dict] = []

    def add(kind: str, label: str, items: list[tuple[str, str]], **extra) -> None:
        for i, (a, b) in enumerate(items, 1):
            row = {"id": f"{kind}-{i:03d}", "a": a, "b": b, "label": label, "type": kind}
            row.update({"user_a": "user-1", "user_b": "user-1"})
            row.update(extra)
            pairs.append(row)

    add("paraphrase", "hit", PARAPHRASE)
    add("reorder", "hit", REORDER)
    add("typo", "hit", [(b, _typo(b, w, bad)) for b, (w, bad) in zip(TYPO_BASES, TYPO_EDITS)])
    add("case", "hit", [(b, b.upper() if i % 2 else b.lower()) for i, b in enumerate(CASE_BASES)])
    add("entity_swap", "miss", ENTITY_SWAP)
    add("number_change", "miss", NUMBER_CHANGE)
    add("negation", "miss", NEGATION)
    add("time_scope", "miss", TIME_SCOPE)
    add("personal", "miss", PERSONAL, user_a="user-1", user_b="user-2")
    add("topic_near", "miss", TOPIC_NEAR)
    return pairs


def main() -> None:
    pairs = build()
    OUT.parent.mkdir(exist_ok=True)
    header = (
        "# Synthetic, fictional question pairs for the false-hit benchmark.\n"
        "# label: hit = reusing the cached answer is acceptable; miss = must not be reused.\n"
        "# Generated by scripts/make_pairs.py (the lists in that file are the source).\n"
    )
    OUT.write_text(header + yaml.safe_dump(pairs, sort_keys=False, allow_unicode=True, width=200))
    adv = [
        {"id": f"adv-{cat}-{i:03d}", "a": a, "b": b, "label": "hit", "type": cat,
         "user_a": "user-1", "user_b": "user-1"}
        for cat, items in ADVERSARIAL.items() for i, (a, b) in enumerate(items, 1)
    ]
    OUT_ADV.write_text(
        "# Held-out adversarial hit pairs (all label: hit), see scripts/make_pairs.py.\n"
        + yaml.safe_dump(adv, sort_keys=False, allow_unicode=True, width=200)
    )
    print("adversarial:", len(adv))
    by: dict[str, int] = {}
    for p in pairs:
        by[p["type"]] = by.get(p["type"], 0) + 1
    print(len(pairs), by)


if __name__ == "__main__":
    main()
