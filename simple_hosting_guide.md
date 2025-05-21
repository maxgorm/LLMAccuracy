# Simple Guide to Host Your Streamlit App for Coworkers

This guide provides straightforward steps to host your Streamlit app so your coworkers can access it.

## Option 1: Host on Your Computer (Simplest for Same Office)

If you and your coworkers are on the same network (same office):

1. Open Command Prompt and run:
   ```
   streamlit run csv_comparison_app.py --server.port=8501 --server.address=0.0.0.0
   ```

2. Find your computer's IP address:
   ```
   ipconfig
   ```
   Look for "IPv4 Address" (typically something like 192.168.1.x)

3. Share the link with coworkers:
   ```
   http://YOUR_IP_ADDRESS:8501
   ```
   For example: `http://192.168.1.100:8501`

4. Keep your computer running while others need to access the app

**Note**: Your computer must stay on and running for the app to be accessible.

## Option 2: Host on a Company Server

If you have access to a company server:

1. Copy your files to the server
2. Install Python and required packages:
   ```
   pip install streamlit pandas openpyxl portkey-ai deepdiff requests pdfplumber tabula-py aiohttp
   ```
   Or use your requirements.txt:
   ```
   pip install -r requirements.txt
   ```

3. Run the app:
   ```
   streamlit run csv_comparison_app.py --server.port=8501 --server.address=0.0.0.0
   ```

4. Share the server's IP address with coworkers:
   ```
   http://SERVER_IP:8501
   ```

5. To keep the app running after you log out, use:
   - On Windows: Set up as a Windows Service
   - On Linux: Use `screen` or `tmux` or set up a systemd service

## Option 3: Use ngrok for Remote Access

If coworkers need to access from outside the office:

1. Install ngrok from [ngrok.com](https://ngrok.com/download)

2. Run your Streamlit app:
   ```
   streamlit run csv_comparison_app.py
   ```

3. In a separate command prompt, run:
   ```
   ngrok http 8501
   ```

4. Ngrok will provide a public URL (like `https://a1b2c3d4.ngrok.io`)

5. Share this URL with your coworkers

**Note**: Free ngrok has limitations (session expires after a few hours)

## Option 4: Use Streamlit's Built-in Sharing Feature

For a temporary solution during development:

1. Run your app:
   ```
   streamlit run csv_comparison_app.py
   ```

2. Click "Network URL" in the terminal output

3. Share this URL with coworkers on the same network

## Making Your App More Robust

For a more permanent solution:

1. **Create a startup script** (Windows example - save as `start_app.bat`):
   ```
   @echo off
   cd C:\path\to\your\app
   streamlit run csv_comparison_app.py --server.port=8501 --server.address=0.0.0.0
   ```

2. **Set up auto-restart** in case of crashes:
   - On Windows: Use Task Scheduler to run the batch file
   - On Linux: Create a systemd service

3. **Consider a simple reverse proxy** like Nginx if you want to:
   - Use a cleaner URL (without the port number)
   - Add basic authentication
   - Enable HTTPS

## Troubleshooting

- **Firewall issues**: Make sure port 8501 is allowed in your firewall
- **Connection refused**: Check if the app is running and the IP address is correct
- **Can't access from outside**: Your network may be blocking external access; consider ngrok
