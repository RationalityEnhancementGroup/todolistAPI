// adapted from https://www.jspsych.org/overview/data/
var user_id = jsPsych.randomization.randomID(8);
jsPsych.data.addProperties({
  subject: user_id
});

// adapted from https://www.jspsych.org/overview/data/
function saveData() {
  var xhr = new XMLHttpRequest();
  xhr.open('POST', '../experiment_data'); // change 'write_data.php' to point to php script.
  xhr.setRequestHeader('Content-Type', 'application/json');
  xhr.send(jsPsych.data.get().json());
}


overview = ["Welcome! In this experiment you will test a to-do list app and engage in goal and task setting.",
     "Please keep this window open while working through the HIT, as it will guide you through the process and give you your final completion code. (It is okay to minimize the window when necessary.)",
     "There will be four parts of the experiment, all with their own bonus:",
     "(1) First you will sign up for our app (don't worry, your real email address is not needed) "+bonus1,
     "(2) You will complete the onboarding process by filling in the goals and tasks you have for the next "+ duration+ " "+bonus2,
     "(3) We will give you some short usability tasks "+bonus3,
     "(4) You will be asked to answer a survey found on the app's website "+bonus4,
     "This should all take around "+exp_duration+". Please press next when you are ready to begin."]


first_step = ["<b>Signing up for the app "+bonus1+"</b>",
        "Please go to this link and sign up for our app: <a href='"+experiment_link+"'>"+experiment_link+"</a>",
        "You do not need to use your own email address. For a temporary email address and password we recommend:",
        "Email address: <a href='"+email_link+"'>"+email_link+"</a><br>Password generator: <a href='"+password_link+"'>"+password_link+"</a>",
        "Please press next when you have finished signing up for the app."
        ]

second_step = ["<b>Onboarding "+bonus2+"</b>,",
        "Once you have signed up for the app you will be instructed to enter your current goals.",
        "After that, you will be asked to sign up for a Workflowy account to input your tasks for each goal. Again, you do not need to use your own credentials:",
        "Email address: <a href='"+email_link+"'>"+email_link+"</a><br>Password generator: <a href='"+password_link+"'>"+password_link+"</a>",
        "Please input your goals and tasks for the next "+duration+"."]

third_step = ["<b>Usability tasks "+bonus3+"</b>",
        "Now that you have inputted your goals and tasks, it's time to test out the experiment a little.",
        "Please press next to get your task."]

usability_one = "Pretend you completed an intention. Check off the intention in the Complice app. Do you have any comments about how this worked?"
usability_two = "Now you remember you forgot an intention you urgently need to complete today. Please enter an intention in Workflowy as part of your first goal, with duration 2 hours and tag it #today. Pull from Workflowy to add the task to your current list. Do you have any comments about how this worked?"

fourth_step = ["<b>Survey "+bonus4+"</b>",
        "Now you are almost done! We just need you to complete a survey (approx "+survey_time+".)",
        "To find the survey please press this button which says " + survey_text +":",
        "<img src=\""+survey_img+"\"",
        "Once you are done please press next to receive your completion code!"]
completion_code = ["Your completion code is: " + user_id,
           "Please enter this code on MTurk. We will bonus you in the next 24 hours!",
           "You can now close this window."]

var instructions_timeline = [{
  type:'external-html',
  url: consent_form_url,
  cont_btn:'consent-to-do-submit-button'
},{
    type: 'instructions',
    pages: [overview.join("<br><br><br>")],
    show_clickable_nav: true
},{
    type: 'instructions',
    pages: [first_step.join("<br><br><br>")],
    show_clickable_nav: true
},{
    type: 'instructions',
    pages: [second_step.join("<br><br><br>")],
    show_clickable_nav: true
},{
    type: 'instructions',
    pages: [third_step.join("<br><br><br>")],
    show_clickable_nav: true
}, {
  type: 'survey-text',
  questions: [
    {prompt: usability_one, rows: 5, columns: 40}, 
    {prompt: usability_two, rows: 5, columns: 40}
  ],
},{
    type: 'instructions',
    pages: [fourth_step.join("<br><br><br>")],
    show_clickable_nav: true
},{
  type: 'call-function',
  func: saveData
},{
    type: 'instructions',
    pages: [completion_code.join("<br><br><br>")],
    show_clickable_nav: true
}]