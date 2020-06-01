import pytest
import json
import requests
from glob import glob
import os 
from pymongo import MongoClient

conn = MongoClient(os.environ['MONGODB_URI'] + "?retryWrites=false")
db = conn.heroku_g6l4lr9d


def large_files_integration():
	def delete_test_records():
		db.request_log.delete_many({ "user_key": "_test_" })
	large_files = glob("data/size/*")
	request_url = "http://aqueous-hollows-34193.herokuapp.com/api/greedy/mdp/45/14/inf/0/inf/0/inf/0/no_scaling/false/0/tree/_test_/getTasksForToday"
	results = {}

	for file in large_files:
		tasks = json.load(open(file,"rb"))
		out = requests.post(request_url, json = tasks)
		results[file] = out

	delete_test_records()
	assert all([out.status_code==200 for file,out in results.items()])
