import os
# from dotenv import load_dotenv, dotenv_values, set_key

DB_HOST = os.getenv('DB_HOST', default='localhost')
DB_PORT = int(os.getenv('DB_PORT', default='27017'))
DB_USER = os.getenv('DB_USER', default='')
DB_PASSWORD = os.getenv('DB_PASSWORD', default='')
DB_AUTH_SOURCE = os.getenv('DB_AUTH_SOURCE', default='')


# dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
# if os.path.exists(dotenv_path):
#     load_dotenv(dotenv_path)

# set_key(dotenv_path, 'DB_HOST', DB_HOST)
# set_key(dotenv_path, 'DB_PORT', str(DB_PORT))
# set_key(dotenv_path, 'DB_USER', DB_USER)
# set_key(dotenv_path, 'DB_PASSWORD', DB_PASSWORD)
# set_key(dotenv_path, 'DB_AUTH_SOURCE', DB_AUTH_SOURCE)

# config = dotenv_values(dotenv_path)

# locals().update(config)
# DB_PORT = int(DB_PORT)


