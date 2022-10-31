from os import cpu_count

# keyfile="/etc/nginx/ssl/file.key"  #optional, commented out pending nginx integration
# certfile="/etc/nginx/ssl/file.cer"  #optional, commented out pending nginx integration
timeout = 500 
workers = 2 * cpu_count() + 1
name = "30day-predict"
bind = "0.0.0.0:5000"  #localhost:port
worker_class = "gthread"  #uvicorn workers for asgi, but flask is wsgi, gthread when deployed in cloud
accesslog="gunicorn_access.log"
wsgi_app = "app:app"
capture_output = False  #captures prints in errorlog
loglevel = "info"  #captures prints in errorlog