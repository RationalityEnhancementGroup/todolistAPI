This is the server for our future Complice experiments with Workflowy task lists.

# Possible Routes (WIP)

None of these have been implemented yet, but we are hoping to create  the following routes (parameters TBD):

* `/static`
* `/random`
* `/hierarchical`
* `/length`
* `/length-deadline`
* `/old-report`

# Required Python Packages

Use requirements.txt


## To Start Server Locally

In the main folder:

```
python app.py
```

## To Test Server

POST http://127.0.0.1:5000/tree/5834b31a714b17bbe10235da520ea3c3162a037e929449aeb6bba2e971efeb79/getTasksForToday

or 

POST https://safe-retreat-20317.herokuapp.com/tree/5834b31a714b17bbe10235da520ea3c3162a037e929449aeb6bba2e971efeb79/getTasksForToday

with the JSON body sample_request.json