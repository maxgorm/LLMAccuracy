# Streamlit App Hosting Guide

This README provides simple instructions for hosting your Streamlit app so your coworkers can access it.

## Quick Start Guide

1. **Start the Streamlit App**:
   - Double-click `start_streamlit_app.bat`
   - This will run your app and make it accessible on your network

2. **Get the Sharing URL**:
   - Right-click `get_sharing_url.ps1` and select "Run with PowerShell"
   - The script will show you the URL to share with your coworkers
   - Example: `http://192.168.1.100:8501`

3. **Share with Coworkers**:
   - Send the URL to your coworkers
   - They can access the app through their web browsers
   - Your computer must stay on with the app running

## Files Included

- `start_streamlit_app.bat` - Batch file to start the Streamlit app
- `get_sharing_url.ps1` - PowerShell script to find your IP address and generate sharing URLs
- `simple_hosting_guide.md` - Detailed guide with more hosting options
- `streamlit_hosting_guide.md` - Advanced guide for cloud hosting options

## Troubleshooting

- **Firewall Issues**: If coworkers can't connect, you may need to allow port 8501 in your firewall
- **App Not Starting**: Make sure all required packages are installed (`pip install -r requirements.txt`)
- **Remote Access**: For access outside your network, see the ngrok option in `simple_hosting_guide.md`

## Keeping the App Running

To keep the app running even when you're not logged in:

1. Open Task Scheduler
2. Create a new task
3. Set the action to start a program
4. Browse to the `start_streamlit_app.bat` file
5. Set it to run with highest privileges
6. Configure additional settings as needed (run at startup, etc.)

## Need More Options?

For more advanced hosting options, including cloud deployment, see `streamlit_hosting_guide.md`.
