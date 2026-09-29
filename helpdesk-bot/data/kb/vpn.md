# VPN (NorthLink)

## When you need the VPN
Northwind uses the NorthLink VPN client. You need it whenever you reach internal tools such as the wiki, build servers and internal dashboards from outside the office. You do not need the VPN for email, Teams or OneDrive. Install NorthLink from Software Center and sign in with your normal work account (single sign-on).

## Common VPN errors
Error 809 means a firewall or router is blocking the VPN traffic. Open NorthLink, go to Settings, then Connection, then Protocol, and switch from UDP to TCP fallback. Hotel and public networks often block UDP. If the VPN connects but is slow, switch to the Northlink-Split profile, which sends only internal traffic through the VPN.

## Session limits
A VPN session lasts at most 8 hours, after which it disconnects automatically. Simply reconnect to start a new session. Unsaved work on internal tools may be lost when the session ends, so save regularly.
