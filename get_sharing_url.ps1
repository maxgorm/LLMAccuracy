# PowerShell script to get the IP address and generate a sharing URL for the Streamlit app

# Get all IP addresses
$ipAddresses = Get-NetIPAddress | Where-Object {$_.AddressFamily -eq "IPv4" -and $_.PrefixOrigin -ne "WellKnown"}

Write-Host "Available IP addresses on this computer:" -ForegroundColor Green
Write-Host ""

# Display all IP addresses
foreach ($ip in $ipAddresses) {
    $interfaceName = (Get-NetAdapter | Where-Object {$_.ifIndex -eq $ip.InterfaceIndex}).Name
    Write-Host "Interface: $interfaceName"
    Write-Host "IP Address: $($ip.IPAddress)"
    
    # Generate the URL for this IP
    $url = "http://$($ip.IPAddress):8501"
    Write-Host "Sharing URL: $url" -ForegroundColor Cyan
    Write-Host ""
}

# Identify the most likely IP address to use (typically the one with a 192.168.x.x or 10.x.x.x pattern)
$likelyIPs = $ipAddresses | Where-Object {$_.IPAddress -match "^(192\.168\.|10\.|172\.1[6-9]\.|172\.2[0-9]\.|172\.3[0-1]\.)"}

if ($likelyIPs) {
    Write-Host "Most likely IP address to share with coworkers:" -ForegroundColor Yellow
    foreach ($ip in $likelyIPs) {
        $url = "http://$($ip.IPAddress):8501"
        Write-Host $url -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "Instructions:" -ForegroundColor Green
Write-Host "1. Make sure the Streamlit app is running (use start_streamlit_app.bat)"
Write-Host "2. Share the appropriate URL with your coworkers"
Write-Host "3. Ensure your firewall allows connections on port 8501"
Write-Host ""
Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
