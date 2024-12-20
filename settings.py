import os
from dotenv import load_dotenv, dotenv_values, set_key

DB_HOST = 'localhost'
DB_PORT = 27017
DB_USER = ''
DB_PASSWORD = ''
DB_AUTH_SOURCE = ''


dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)

set_key(dotenv_path, 'DB_HOST', DB_HOST)

config = dotenv_values(dotenv_path)

locals().update(config)
DB_PORT = int(DB_PORT)


