const fileInput = document.getElementById('fileInput');
const analyzeBtn = document.getElementById('analyzeBtn');
const filePill = document.getElementById('filePill');
const fileName = document.getElementById('fileName');

fileInput.addEventListener('change', () => {
  const file = fileInput.files[0];
  if(file){
    analyzeBtn.disabled = false;
    filePill.classList.add('show');
    fileName.textContent = file.name;
  }
});

window.analyzeResume = async function(){

  const file = fileInput.files[0];
  if(!file){
    alert("Select file");
    return;
  }

  const resultCard = document.getElementById("resultCard");
  const desc = document.getElementById("gradeDesc");
  const list = document.getElementById("suggestList");
  const suggestionsBox = document.getElementById("suggestionsBox");
  const scoreCircle = document.getElementById("scoreCircle");

  resultCard.classList.remove("hidden");

  document.getElementById("scoreValue").innerText = "--";
  document.getElementById("gradeTitle").innerText = "Processing...";
  desc.textContent = "Starting analysis...";
  list.innerHTML = "";

  suggestionsBox.classList.add("hidden");
  scoreCircle.classList.add("hidden");

  const fd = new FormData();
  fd.append("resume", file);

  // 🔥 STEP MESSAGES
  const steps = [
    "Extracting text from PDF…",
    "Detecting skills with NLP…",
    "Scoring GitHub repositories…",
    "Running XGBoost classifier…",
    "Finalising grade…"
  ];

  let i = 0;

  const interval = setInterval(()=>{
    if(i < steps.length){
      desc.textContent = steps[i];
      i++;
    }
  }, 1050);  // speed of change

  try{
    const res = await fetch("/predict",{
      method:"POST",
      body:fd
    });

    const data = await res.json();

    clearInterval(interval); // 🔥 STOP steps

    const grade = data.grade || "D";
    const scoreMap = {A:90,B:75,C:60,D:40};
    const score = scoreMap[grade];

    scoreCircle.classList.remove("hidden");

    const circle = document.getElementById("progressCircle");
    circle.style.strokeDashoffset = 314;

    setTimeout(()=>{
      const offset = 314 - (314 * score)/100;
      circle.style.strokeDashoffset = offset;
    },200);

    document.getElementById("scoreValue").innerText = score;
    document.getElementById("gradeTitle").innerText = "Grade: "+grade;
    desc.textContent = "Analysis completed";

    suggestionsBox.classList.remove("hidden");

    let suggestions = [];

    if(score >= 85){
      suggestions = ["Strong resume","Add leadership","Tailor for roles"];
    }
    else if(score >= 70){
      suggestions = ["Add projects","Improve GitHub","Use numbers"];
    }
    else if(score >= 50){
      suggestions = ["Improve skills","Fix formatting","Add projects"];
    }
    else{
      suggestions = ["Build projects","Learn basics","Rewrite resume"];
    }

    suggestions.forEach(s=>{
      const li = document.createElement("li");
      li.textContent = s;
      list.appendChild(li);
    });

  }catch(err){
    clearInterval(interval); // 🔥 stop even on error
    console.error(err);
    alert("Error");
  }
};

window.resetAll = function(){
  location.reload();
};