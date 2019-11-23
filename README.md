This is the server for our future Complice experiments with Workflowy task lists.

# Possible Routes

The following point assigners will be implemented:

* `/constant`
* `/random`
* `/hierarchical`
* `/length`
* `/old-report`

The following scheduler APIs will also be implemented:

* `/basic`
* `/deadline`

A possible route is therefore:
`/api/length/basic/tree/5834b31a714b17bbe10235da520ea3c3162a037e929449aeb6bba2e971efeb79/getTasksForToday`
or 
`/api/random/deadline/100/2/tree/5834b31a714b17bbe10235da520ea3c3162a037e929449aeb6bba2e971efeb79/getTasksForToday`

# Required Python Packages

Use requirements.txt


## To Start Server Locally

In the main folder:

```
python app.py
```

## To Test Server

Use JSON body sample_request.json

Locally:

```
POST http://127.0.0.1:5000/[request]
```

or 

On heroku:

```
POST https://safe-retreat-20317.herokuapp.com/[request]

```
