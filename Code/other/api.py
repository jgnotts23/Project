#!/usr/bin/env python3

""" """

__appname__ = 'api.py'
__author__ = 'Jacob Griffiths (jacob.griffiths18@imperial.ac.uk)'
__version__ = '0.0.1'

## Imports ##

from boxsdk import Client, OAuth2

auth = OAuth2(
  client_id='4pnxzx0n07tvdi95zg77mxr4ngax4o08',
  client_secret='LUbfOvc9xU0InEecFrFX7D5CvOpDGc2y',
  access_token='0PQPNnmgbxIWqxVAUs2xQfjKPAi6qiRb',
)


client = Client(auth)

user = client.user().get()


root_folder = client.folder(folder_id='0').get()
folder_items = root_folder.get_items(limit=100,offset=0)

SQ283 = client.folder(folder_id='76788541248').get()




for folder_item in folder_items:
  if folder_item['name'] == 'SQ283' :
   folder = client.folder(folder_item['id']).get()
   break



# do 2 files
file_list = ['file1','file2']
for file_name in file_list :
  box_file = folder.upload(file_path=file_name)
