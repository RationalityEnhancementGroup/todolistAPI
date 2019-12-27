
function create_questionnaire(questions, scale, preamble){
	var questions_formatted = [];
	for (i = 0; i < questions.length; i++) {
		  questions_formatted.push({prompt:questions[i], labels:scale, required:false})
		};
	return {type: 'survey-likert',
    		questions: questions_formatted,
    		preamble: preamble}
    		};

var survey_timeline = [];
var num_goals = 3;
var curr_goal = 1;

survey_timeline.push({type: 'survey-likert',
    preamble: 'Please answer the following questions about your experience with using the to-do list website.',
    questions: [{prompt: "How many goals did you enter on the CompliceX app?", labels: ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "10+"],required:false}],
    on_finish: function(data) {num_goals = JSON.parse(data.responses)["Q0"]+1}
});


var cluster_preamble = "Some people have goals that are very separate, while others may have goals that fall into the same category.<br>"+
"For example, you may have multiple goals for making progress at work or furthering your education. Or perhaps, you have multiple goals for how to improve your health. <br>"+
"Please think about whether there are any larger goals that subsume multiple of the goals you entered in WorkFlowy. For instance, if your first and your third WorkFlowy goal are both part of some larger goal, then please assign both of them to Larger Goal 1."+ 
"And if on top of that your second and fourth goal are part of another larger goal, then please assign both of them to Larger Goal 2."

var cluster_questions = ["Goal 1", "Goal 2", "Goal 3", "Goal 4", "Goal 5", "Goal 6", "Goal 7", "Goal 8", "Goal 9", "Goal 10"]
var cluster_labels = ["Larger Goal 1", "Larger Goal 2", "Larger Goal 3", "Larger Goal 4", "Larger Goal 5", "Larger Goal 6", "Larger Goal 7", "Larger Goal 8", "Larger Goal 9", "Larger Goal 10"]


// survey_timeline.push(create_questionnaire(questions, labels, cluster_preamble, num_goals));
survey_timeline.push({type: 'survey-likert',
    		questions: function(){var questions_formatted = [];
							for (i = 0; i < num_goals; i++) {
								  questions_formatted.push({prompt:cluster_questions[i], labels:cluster_labels.slice(0, num_goals), required:false})
								}; return questions_formatted;},
    		preamble: cluster_preamble}
    		);

var SIMS_Scale = ["corresponds not at all",
"corresponds a very little",
"corresponds a little",
"corresponds moderately",
"corresponds enough",
"corresponds a lot",
"corresponds exactly"]



var SIMS_Questions = ["Because I think that working on this goal is interesting",
"Because I am doing it for my own good",
"Because I am supposed to do it",
"There may be good reasons to work on this goal, but personally I don’t see any",
"Because I think that working on this goal is pleasant",
"Because I think that working on this goal is good for me",
"Because it is something that I have to do",
"I work on this goal but I am not sure if it is worth it",
"Because working on this goal is fun",
"By personal decision",
"Because I don’t have any choice",
"I don’t know; I don’t see what working on this goal brings me",
"Because I feel good when working on this goal",
"Because I believe that working on this goal is important for me",
"Because I feel that I have to do it",
"I work on this goal, but I am not sure it is a good thing to pursue it"]


// survey_timeline.push({
// 	timeline: [{
// 		type: 'survey-likert',
// 		preamble: function(){return "<b>Goal "+curr_goal+"</b><br>Why are you currently engaged in this goal? Please read each item carefully and respond with how much each statement corresponds with your reasons for engaging in this goal."},
// 		questions: function(){var questions_formatted = [];
// 							for (i = 0; i < SIMS_Questions.length; i++) {
// 		  						questions_formatted.push({prompt:SIMS_Questions[i], labels:SIMS_Scale, required:false})}; 
// 								return questions_formatted;}
// 	}],
	// loop_function: function(data){
	// 	curr_goal = curr_goal+1;
	// 	console.log(curr_goal)
	// 	if (curr_goal > num_goals){
	// 		return false;
	// 	} else {
	// 		return true;
	// 	}}})

survey_timeline.push({
		type: 'survey-likert',
		preamble: function(){return "<b>Goal 1 (Most Valuable Goal in Workflowy)</b><br>Why are you currently engaged in this goal? Please read each item carefully and respond with how much each statement corresponds with your reasons for engaging in this goal."},
		questions: function(){var questions_formatted = [];
							for (i = 0; i < SIMS_Questions.length; i++) {
		  						questions_formatted.push({prompt:SIMS_Questions[i], labels:SIMS_Scale, required:false})}; 
								return questions_formatted;}})
survey_timeline.push({
		type: 'survey-likert',
		preamble: function(){return "<b>Goal 2 (Second Most Valuable Goal in Workflowy)</b><br>Why are you currently engaged in this goal? Please read each item carefully and respond with how much each statement corresponds with your reasons for engaging in this goal."},
		questions: function(){var questions_formatted = [];
							for (i = 0; i < SIMS_Questions.length; i++) {
		  						questions_formatted.push({prompt:SIMS_Questions[i], labels:SIMS_Scale, required:false})}; 
								return questions_formatted;}})

scale_R = ['not at all','','','','moderately','','','','very much']
scale_M = ['not at all','','','','moderately','','','','very much']
var reward = {
    preamble: 'Below is a list of statements about how you might or might not have experienced the daily intentions on the to-do list website. For each statement please indicate how much you agree with it.',
    type: 'survey-likert',
    questions: [{prompt: 'It felt rewarding to complete a daily intention.', labels: scale_R, required: false},
                {prompt: 'It felt good to have finished a daily intention.', labels: scale_R, required: false},
                {prompt: 'When I submitted a daily intention I felt like I had accomplished something.', labels: scale_R, required: false},
                {prompt: 'Completing a daily intention felt like progress.', labels: scale_R, required: false},
                {prompt: 'I experienced submitting each daily intention as a success.', labels: scale_R, required: false}]
}
var motivation = {
    preamble: 'Below is a list of statements about how you might or might not have experienced the daily intentions. For each statement please indicate how much you agree with it.',
    type: 'survey-likert',
    questions: [{prompt: 'I felt motivated cto omplete the daily intentions.', labels: scale_M, required: false},
                {prompt: 'It was really hard for me to get myself to work on the tasks on my to-do list.', labels: scale_M, required: false},
                {prompt: 'I did <u>not</u> feel like working on the tasks on my to-do list.', labels: scale_M, required: false},
                {prompt: 'Getting started on the tasks on my to-do list was easy for me.', labels: scale_M, required: false}]
}


// survey_timeline.push(reward)
// survey_timeline.push(motivation)

var text_survey = {
    type: 'survey-text',
    questions: [{prompt: "How old are you?"},
                {prompt: "Which gender do you identify with?"},
                {prompt: "Which features of the to-do list website did you find helpful?"},
                {prompt: "Do you have any further comments?"}
                ]

}



//survey_timeline.push(divider)

            
// defining two different response scales that can be used.
var scale_1 = ["Not at all", "", "", "", "","","","","Completely"];
questions=["To what extent did you <u>feel in control</u> of your choice which daily intention to work on?",
           "To what extent did you feel your choice to be <u>thought-through</u>?",
           "To what extent did you feel that your choice <u>belonged to you</u>?",
           "To what extent did you feel that your choice <u>reflected your preferences</u>?",
           "To what extent do you feel that you can <u>\"stand for\"</u> your choice?",
           "To what extent do you feel that the choice you ended up making was <u>free from external influence</u>?"
          ]            
var autonomy_scale = {
    type: 'survey-likert',
    preamble: 'Please answer the following questions about your experience with using the to-do list website.',
    questions: [{prompt: questions[0], labels: scale_1,required:false},
                {prompt: questions[1], labels: scale_1,required:false},
                {prompt: questions[2], labels: scale_1,required:false},
                {prompt: questions[3], labels: scale_1,required:false},
                {prompt: questions[4], labels: scale_1,required:false},
                {prompt: questions[5], labels: scale_1,required:false},
                ]
};
// survey_timeline.push(autonomy_scale)


// defining two different response scales that can be used.
var scale_2 = ["Do <b>not</b> agree at all", "", "", "", "","","","","Agree completely"];
intrusion_questions=["The to-do list website <u>threatened my freedom to choose what I wanted</u>.",
           "The to-do list website <u>tried to make decisions for me</u>.",
           "The to-do list website <u>tried to manipulate me</u>.",
           "The to-do list website <u>tried to pressure me</u>."
          ]            
var intrusion_scale = {
    type: 'survey-likert',
    questions: [{prompt: intrusion_questions[0], labels: scale_2,required:false},
                {prompt: intrusion_questions[1], labels: scale_2,required:false},
                {prompt: intrusion_questions[2], labels: scale_2,required:false},
                {prompt: intrusion_questions[3], labels: scale_2,required:false}
                ]
};
// survey_timeline.push(intrusion_scale)


var IPS_preamble = "Please read the following statements carefully and select the response that best describe how you feel about that statement."

var IPS_Scale = [
"Very Seldom or Not True of Me",
"Seldom True of Me",
"Sometimes True of Me",
"Often True of Me",
"Very Often True, or True of Me"]

var IPS_2010 = [
"I put things off so long that my well-being or efficiency unnecessarily suffers.",
"If there is something I should do, I get to it before attending to lesser tasks.",
"My life would be better if I did some activities or tasks earlier.",
"When I sIhould be doing one thing, I will do another.",
"At the end of the day, I know I could have spent the time better.",
"I spend my time wisely.",
"I delay tasks beyond what is reasonable.",
"I procrastinate.",
"I do everything when I believe it needs to be done."]

survey_timeline.push(create_questionnaire(IPS_2010, IPS_Scale, IPS_preamble));

var WE_preamble = "The following 17 statements are about how you feel at work. Please read each statement carefully and decide if you ever feel this way about your job. If you have never had this feeling, mark 'Never'. If you have had this feeling, select the answer that best describes how frequently you feel that way."

var WE_scale = ["Never",
"Almost Never / A few times a year or less",
"Rarely / Once a month or less",
"Sometimes / A few times a month",
"Often / Once a week",
"Very Often / A few times a week",
"Always / Every day"]

var WE_questions=[
"At my work, I feel bursting with energy.",
"I find the work that I do full of meaning and purpose.",
"Time flies when I am working.",
"At my job, I feel strong and vigorous.",
"I am enthusiastic about my job.",
"When I am working, I forget everything else around me.",
"My job inspires me.",
"When I get up in the morning, I feel like going to work.",
"I feel happy when I am working intensely.",
"I am proud of the work that I do.",
"I am immersed in my work.",
"I can continue working for very long periods at a time.",
"To me, my job is challenging.",
"I get carried away when I am working.",
"At my job, I am very resilient, mentally.",
"It is difficult to detach myself from my job.",
"At my work, I always persevere, even when things do not go well."]

survey_timeline.push(create_questionnaire(WE_questions, WE_scale, WE_preamble));

var NGSE_preamble = "Please read the following statements carefully and select the response that best describe how you feel about that statement."

var NGSE_Scale = [
"Strongly agree",
"Agree",
"Neither agree nor disagree", 
"Disagree",
"Strongly disagree"]

var NGSE_Questions =["I will be able to achieve most of the goals that I set for myself.",
"When facing difficult tasks, I am certain that I will accomplish them.",
"In general, I think that I can obtain outcomes that are important to me.",
"I believe I can succeed at most any endeavor to which I set my mind.",
"I will be able to successfully overcome many challenges.",
"I am confident that I can perform effectively on many different tasks.",
"Compared to other people, I can do most tasks very well.",
"Even when things are tough, I can perform quite well."]

survey_timeline.push(create_questionnaire(NGSE_Questions, NGSE_Scale, NGSE_preamble));

var SWLS_preamble = "Please read the following statements carefully and select the response that describes best how you feel about that statement."

var SWLS_Scale = [
"Strongly agree",
"Agree",
"Slightly agree",
"Neither agree nor disagree", 
"Slightly disagree",
"Disagree",
"Strongly disagree"]

var SWLS_Questions = [
"In most ways my life is close to my ideal.", 
"The conditions of my life are excellent.",
"I am satisfied with my life.",
"So far I have gotten the important things I want in life.",
"If I could live my life over, I would change almost nothing."]

survey_timeline.push(create_questionnaire(SWLS_Questions, SWLS_Scale,SWLS_preamble));

var grit_preamble = "Please read the following statements carefully and select the response that describes best how you feel about that statement."

var grit_scale = [
"Very much like me",
"Mostly like me",
"Somewhat like me",
"Not much like me",
"Not like me at all"]

var short_grit = [
"New ideas and projects sometimes distract me from previous ones.",
"Setbacks don’t discourage me.",
"I have been obsessed with a certain idea or project for a short time but later lost interest.",
"I am a hard worker.",
"I often set a goal but later choose to pursue a different one.",
"I have difficulty maintaining my focus on projects that take more than a few months to complete.",
"I finish whatever I begin.",
"I am diligent."
]

survey_timeline.push(create_questionnaire(short_grit, grit_scale,grit_preamble));

var occupation_options = ["Self-employed", "Skilled laborer", "Student", "Researcher/Scientist", "Administrative staff", "Salesperson", "Manager", "Educator", "Other - Knowledge Worker", "Other"]

var occupation = {
  type: 'survey-multi-select',
  questions: [
    {prompt: "Which of the following best describes your occupation? Please check all that apply.", name: 'occupation', options: occupation_options, required:false}, 
  ],
};

survey_timeline.push(occupation)
survey_timeline.push(text_survey)


