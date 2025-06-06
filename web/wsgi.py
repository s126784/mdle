import sys
import os

project_home = '/home/solova88/mysite/web'
if project_home not in sys.path:
    sys.path = [project_home] + sys.path

from app import app as application

if __name__ == "__main__":
    application.run()
