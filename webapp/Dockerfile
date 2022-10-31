FROM python:3.7-slim
WORKDIR /app

# Note:  this docker file assumes it is running in the context of the repo root, though Docker file is 
# two level down in code/webapp folder.   Build yaml sets the context and docker file.

# Copy just requirements so we don't have to re-do pip install every time code changes
# then run pip install
# TODO: pin reqs
COPY /webapp/requirements.txt /app/requirements.txt
RUN pip install  --no-cache-dir -r /app/requirements.txt

# Copy the rest of the code....skipping directories that may contain data
COPY .. /app/

# set python path and working dir 
ENV PYTHONPATH=/app/
WORKDIR /app/webapp

# run the flask app!
# host must be 0.0.0.0 for cloud run to pick it up.
ENTRYPOINT ["sh", "-x", "gunicorn.sh"]