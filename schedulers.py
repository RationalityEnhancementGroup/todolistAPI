def basic_scheduler(projects, today_duration=8*60, with_today = True):
	'''
	Takes in flattened project tree with "reward" from some API
	Outputs list of tasks for today
	'''
	duration_remaining = today_duration

	task_list = []
	for goal in projects:
		task_list.extend(goal["ch"])

	if with_today == False:
		final_tasks = []
	else:
		final_tasks = [task for task in task_list if task["today"] == 1]
		duration_remaining -= sum([task["est"] for task in final_tasks])

	sorted_by_value = sorted(task_list, key=lambda k: k['val']) #from: https://stackoverflow.com/a/73050
	for task in sorted_by_value[::-1]:
		if task["est"] <= duration_remaining:
			final_tasks.append(task)
			duration_remaining -= task["est"]

	return final_tasks
