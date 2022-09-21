import * as DICTIONARY from './dictionary.js';

// const MODEL_JSON_URL = './spam_model/tfjs/model.json';
const MODEL_JSON_URL = './comment_model/tfjs/model.json';
const SPAM_THRESHOLD = 0.75;
const ENCODING_LENGTH = 200;
var model;

function tokenize(wordArray) {
  // Always start with the START token.
  let returnArray = [DICTIONARY.START];
  
  // Loop through the words in the sentence you want to encode.
  // If word is found in dictionary, add that number else
  // you add the UNKNOWN token.
  for (var i = 0; i < wordArray.length; i++) {
    let encoding = DICTIONARY.LOOKUP[wordArray[i]];
    returnArray.push(encoding === undefined ? DICTIONARY.UNKNOWN : encoding);
  }
  
  // Finally if the number of words was < the minimum encoding length
  // minus 1 (due to the start token), fill the rest with PAD tokens.
  while (i < ENCODING_LENGTH - 1) {
    returnArray.push(DICTIONARY.PAD);
    i++;
  }
  
  // Log the result to see what you made.
  // console.log([returnArray]);
  
  // Convert to a TensorFlow Tensor and return that.
  return tf.tensor([returnArray]);
}

async function run(messages) {
  if (model == null) {
    model = await tf.loadLayersModel(MODEL_JSON_URL);
  }
  messages.forEach(async (message) => {
    var msgArray = message.toLowerCase().replace(/[^\w\s]/g, ' ').split(' ');
    var result = await model.predict(tokenize(msgArray));
    result.data().then((dataArray)=>{
      console.log(message, dataArray[1])
    })
  });

}

var reviews = [
  "Your package is waiting for delivery. Please confirm the settlement of $19.99 on the following link: http://aka.ms/adfuyiwy",
  "NHS: we have identified that you are eligible to apply for your vaccine. For more information and apply, follow here: application-ukform.com",
  "I've finally arrived to Vancouver. Already found the office",
  "watching Project Volterra a DevBox / Mini Gaming PC and other cool stuff from MS Build: https://www.youtube.com/watch?v=yICVNta8jMU",
  "It's June and one of my neighbors still have Christmas tree on their deck",
  "What's the difference between the red and the black coat? Is it like belt colors in karate?",
  "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"
]

run(reviews)