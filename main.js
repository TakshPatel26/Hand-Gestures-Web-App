prediction = "";
Webcam.set({
    width: 350,
    height: 300,
    image_format: 'png',
    png_quality: 90
});
camera = document.getElementById("camera");
Webcam.attach('#camera');
function take_snapshot(){
    Webcam.snap(function (data_uri){
        document.getElementById("result").innerHTML = '<img id="image_captured" src="'+data_uri+'"/>';
    });
}
console.log("ml5 version:",ml5.version);
classifier = ml5.imageClassifier('https://teachablemachine.withgoogle.com/models/GWA2DTRFL/model.json',modelLoaded);
function modelLoaded(){
    console.log("Model Loaded Successfully");
}
function speak(){
    var synth = window.speechSynthesis;
    var speak_data = "The prediction is "+prediction;
    var utterThis = new SpeechSynthesisUtterance(speak_data);
    synth.speak(utterThis);
}
function check(){
     img=document.getElementById("image_captured");
     classifier.classify(img,gotResults);
}
function gotResults(error,results){
     if(error){
          console.error(error);
     }
     else{
          console.log(results);
          document.getElementById("result_gesture_name").innerHTML=results[0].label;
          prediction=results[0].label;
          speak();
          if(results[0].label=="Amazing"){
               document.getElementById("result_emoji").innerHTML="&#128076;";
          }
          if(results[0].label=="Best"){
               document.getElementById("result_emoji").innerHTML="&#128077;";
          }
          if(results[0].label=="One"){
               document.getElementById("result_emoji").innerHTML="&#9757;";
          }
          if(results[0].label=="Hand Raise"){
               document.getElementById("result_emoji").innerHTML="&#9995;";
          }
          }
          if(results[0].label=="Swag"){
               document.getElementById("result_emoji").innerHTML="&#129304;";
          }
     }

}
