// adapted from https://www.jspsych.org/overview/data/
var user_id = jsPsych.randomization.randomID(8);
var survey_id = jsPsych.randomization.randomID(8);
jsPsych.data.addProperties({
  subject: user_id,
  survey_id: survey_id
});

// adapted from https://www.jspsych.org/overview/data/
function saveData() {
  var xhr = new XMLHttpRequest();
  xhr.open('POST', '../experiment_data'); // change 'write_data.php' to point to php script.
  xhr.setRequestHeader('Content-Type', 'application/json');
  xhr.send(jsPsych.data.get().json());
}

var check_consent = function(elem) {
  all_checked = (document.thirdConsent.thirdConsentRadios.value==="yes")&&(document.secondConsent.secondConsentRadios.value==="yes")&&(document.firstConsent.firstConsentRadios.value==="yes");
  if (all_checked) {
    return true;
  }
  else {
    alert("If you wish to participate, you must agree with and check the checkboxes above.");
    return false;
  }
  return false;
};



overview = ["Welcome! In this experiment you will test a to-do list app and engage in goal and task setting.",
     "Please keep this window open while working through the HIT, as it will guide you through the process and give you your final completion code. (It is okay to minimize the window when necessary.)",
     "There will be four parts of the experiment, all with their own bonus:",
     "(1) First you will sign up for our app (don't worry, your real email address is not needed) "+bonus1+"<br>(2) You will complete the onboarding process by filling in the goals and tasks you have for the next "+ duration+ " "+bonus2+"<br>(3) We will give you some short usability tasks "+bonus3+"<br>(4) You will be asked to answer a survey found on the app's website "+bonus4,
     "This should all take around "+exp_duration+". Please press next when you are ready to begin."]


first_step = ["<b>Signing up for the app "+bonus1+"</b>",
        "Please go to this link and sign up for our app: <a href='"+experiment_link+"' target=\"_blank\">"+experiment_link+"</a>",
        "You do not need to use your own email address (although it is also fine and possibly easier to do so.) For a temporary email address and password we recommend:",
        "Email address: <a href='"+email_link+"' target=\"_blank\">"+email_link+"</a><br>Password generator: <a href='"+password_link+"' target=\"_blank\">"+password_link+"</a>",
        "You will be asked to enter this code so we can trace back your responses and bonus you correctly: <b>"+survey_id+"</b> (please note: this is <b>not</b> your completion code!)",
        "When you are done signing up you will see this screen:",
        "<img border=\"5\" src=\""+signup_img+"\"",
        "Please press next when you have finished signing up for the app."
        ]

second_step = ["<b>Onboarding "+bonus2+"</b>,",
        "Once you have signed up for the app you will be instructed to enter your current goals.",
        "After that, you will be asked to sign up for a Workflowy account to input your tasks for each goal. Again, you do not need to use your own credentials. You can use the credentials from the last step, or if needed, create new ones:",
        "Email address: <a href='"+email_link+"' target=\"_blank\">"+email_link+"</a><br>Password generator: <a href='"+password_link+"' target=\"_blank\">"+password_link+"</a>",
        "Please input your goals and tasks for the next "+duration+".",
        "Once you are done, you will see:",
        "<img border=\"5\" src=\""+goal_img+"\"",
        "Please press next when you have finished entering your goals and tasks."]

third_step = ["<b>Usability tasks "+bonus3+"</b>",
        "Now that you have inputted your goals and tasks, it's time to test out the experiment a little.",
        "Please try to complete the tasks we give you, but do not worry if you are not able to complete them -- you will not be penalized. We are just looking for effort. Let us know where you were confused, if you do get stuck.",
        "Please press next to get your tasks."]

scale_completion = ["Yes, easily", "Yes, with some experimentation or research", "I'm not sure", "No, I wasn't able to complete the task", "No, not at all"]
scale_binary = ["Yes", "No"]
scale_feeling = ["Extremely","Very","Moderately","Slightly","Not at all"]
frequency_likert = ["Almost always", "Often", "Sometimes", "Seldom", "Never"]
future_action =["Definitely","Probably","Possibly","Probably Not","Definitely Not"]

ux_question_one = {timeline:[{
type: 'survey-likert',
preamble: "<b>Task 1</b>, Page 1 of 2<br>"+"Imagine you just completed an intention. Check off the intention in the CompliceX app.",
questions : [
{prompt: "Were you able to complete this task?" , labels: scale_completion, required: false},
{prompt: "Was this intuitive?", labels: scale_feeling, required: false},
{prompt: "Did you find this action rewarding?", labels: scale_feeling, required: false}]},
{type: 'survey-text',
preamble: "<b>Task 1</b>, Page 2 of 2<br>"+"Imagine you just completed an intention. Check off the intention in the CompliceX app.",
questions: [{prompt: "Do you have any comments or suggestions about how this worked?", rows: 5, columns: 80}]}]}

ux_question_two = {timeline:[{
type: 'survey-likert',
preamble: "<b>Task 2</b>, Page 1 of 2<br>"+"Now imagine you remember you forgot an intention you urgently need to complete today. Please enter an intention in Workflowy as part of your first goal, with duration 2 hours, marked for today. After that, add the task to your intentions list.",
questions:[
{prompt: "Were you able to complete this task?" , labels: scale_completion, required: false},
{prompt: "Was this intuitive?", labels: scale_feeling, required: false},
{prompt: "Can you see yourself adding tasks like this if CompliceX was your daily to-do list app?", labels:future_action, required:false}]},
{type: 'survey-text',
preamble: "<b>Task 2</b>, Page 2 of 2<br>"+ "Now imagine you remember you forgot an intention you urgently need to complete today. Please enter an intention in Workflowy as part of your first goal, with duration 2 hours, marked for today. After that, add the task to your intentions list.",
questions: [{prompt: "Do you have any comments or suggestions about how this worked?", rows: 5, columns: 80}]}]}


ux_question_three = {timeline:[{
type: 'survey-likert',
preamble: "<b>Task 3</b>, Page 1 of 2<br>"+"Currently the app asks for your typical working hours and planned working hours today. This is because we are developing an algorithm that helps users to complete goals, and it will need this information to plan."
+"<br>We have two options we are considering:<br> <b>Option 1</b>: users input typical and today hours, along with whether they work on weekends"
+"<br><b>Option 2</b>: users input hours for every day of the week, and change them as their availability that week changes",
questions:[
{prompt:"Which option would you prefer as a user?", labels: ["Option 1", "Option 2"], required: false},
{prompt:"Do your working hours (towards your inputted goals, not just work or school) differ between weekdays and weekends?", labels: frequency_likert, required: false},
{prompt:"Do your goal working hours differ between days?", labels: frequency_likert, required:false},
{prompt:"Do you typically have a good idea of how many hours you have available for working towards your goal, one week ahead?", labels: frequency_likert, required:false}]},
{type: 'survey-text',
preamble: "<b>Task 3</b>, Page 2 of 2<br>"+"Currently the app asks for your typical working hours and planned working hours today. This is because we are developing an algorithm that helps users complete goals, and it will need this information to plan."
+"<br>We have two labels we are considering:<br> <b>Option 1</b>: users input typical and today hours, along with whether they work on weekends"
+"<br><b>Option 2</b>: users input hours for every day of the week, and change them as their availability that week changes",
questions: [{prompt: "Do you have any additional comments or suggestions for us about asking for working hours?", rows: 5, columns: 80}]}]}

general_usability = {type: 'survey-text',
preamble: "If you were able to complete your tasks quickly, feel free to play around with the app now. (Please note, this is not required.) <br> Feel free to respond to any of the following optional questions for which you have an opinion.",
questions:[
{prompt:"In general, did the to-do list app do anything unexpected or confusing? If so, what was it?", rows: 5, columns: 80},
{prompt:"On a scale from -10 (it was horrible) to +10 (it was amazing) how much did you enjoy using the to-do list app?", rows: 5, columns: 80},
{prompt:"Do you have any comments or suggestions on how we might improve the to-do list app?", rows: 5, columns: 80},
{prompt:"What did you like about the to-do list app?", rows: 5, columns: 80},
{prompt:"Was anything unintuitive? What would you like better explained?", rows: 5, columns: 80}]}

fourth_step = ["<b>Survey "+bonus4+"</b>",
        "Now you are almost done! We just need you to complete a final survey (approx "+survey_time+".)",
        "Press next to begin"]

completion_code = ["Thank you, you are now finished with the experiment.",
          "Your completion code is: " + user_id,
           "Please enter this code on MTurk. We will bonus you within the next 24 hours!",
           "You can now close this window."]

var instructions_timeline = [{
  type:'external-html',
  url: consent_form_url,
  check_fn: check_consent,
  cont_btn:'consent-to-do-submit-button'
},{
    type: 'instructions',
    pages: [overview.join("<br><br><br>")],
    show_clickable_nav: true
},{
    type: 'instructions',
    pages: [first_step.join("<br><br><br>")],
    allow_backward:true,
    show_clickable_nav: true
},{
    type: 'instructions',
    pages: [second_step.join("<br><br><br>")],
    allow_backward:true,
    show_clickable_nav: true
},{
    type: 'instructions',
    pages: [third_step.join("<br><br><br>")],
    allow_backward:true,
    show_clickable_nav: true
},
ux_question_one,
ux_question_two,
ux_question_three,
general_usability,{
    type: 'instructions',
    pages: [fourth_step.join("<br><br><br>")],
    show_clickable_nav: true
},{
  timeline: survey_timeline
},{
  type: 'call-function',
  func: saveData
},{
    type: 'instructions',
    pages: [completion_code.join("<br><br><br>")],
    allow_backward:true,
    show_clickable_nav: true
}]