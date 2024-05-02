var cb = document.getElementById("colorblind");
var lv = document.getElementById("lowvision");
var og = document.getElementById("original");


function colorblind(){
	if (cb.className == "top"){
		document.body.setAttribute("style", "-webkit-filter:grayscale(100%); backdrop-filter:grayscale(100%); background: #C40");
		document.querySelector(".topBar").setAttribute("style", "-webkit-filter: grayscale(100%);");
		document.querySelector(".coloredText1").setAttribute("style", "-webkit-filter: grayscale(100%);");
		var sth = document.querySelector(".top-selected");
		sth.className = "top";
		cb.className = "top-selected";
	}
	else{
		return;
	}
}

$("#colorblind").click(colorblind);

function lowvision(){
	if (lv.className == "top"){
		document.body.setAttribute("style", "-webkit-filter: blur(3px) ;background: #C40");
		//document.querySelector(".topBar").setAttribute("style", "-webkit-filter: grayscale(0%);");
		document.querySelector(".coloredText1").setAttribute("style", "-webkit-filter: grayscale(0%);");
		var sth = document.querySelector(".top-selected");
		sth.className = "top";
		lv.className = "top-selected";
	}
	else{
		return;
	}
}

$("#lowvision").click(lowvision);

function original(){
	if (og.className == "top"){
		document.body.setAttribute("style", "-webkit-filter:blur(0px); background: #C40;");
		document.querySelector(".coloredText1").setAttribute("style", "-webkit-filter: grayscale(0%);");
		var sth = document.querySelector(".top-selected");
		sth.className = "top";
		og.className = "top-selected";
	}
	else{
		return;
	}
}

$("#original").click(original);