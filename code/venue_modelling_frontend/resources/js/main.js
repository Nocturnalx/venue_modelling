var xLength;
var yLength;
var zLength;
var resolution;
var order;

var xNeg;
var xPos;
var yNeg;
var yPos;
var zNeg;
var zPos;

var inputBox;
var inputShowing = false;
var queueID = -1;
var username;
var loggedIn = false;
var downloadReady = false;

var xLeng_vis;
var yLeng_vis;
var zLeng_vis;

var visuFile;

var boxSize;
var currentFrame = 0;
var frameCount;
var frameLeng;
var intervalID;

var topFrames;
var breadthFrames;
var lengthFrames;

var topCtx;
var breadthCtx;
var lengCtx;

var headerSize = 20;

function onload(){
    //floating input box
    //input = document.getElementById('inputBox');

    
}

function newUser()
{
    let input = document.getElementById('usernameInput');
    let loginArea = document.getElementById('loginArea');
    username = input.value;
    if (username != '') {
        let xmlHttp = new XMLHttpRequest();
        xmlHttp.onreadystatechange = function() { 
            if (xmlHttp.readyState == 4 && xmlHttp.status == 200){
                let response = parseInt(xmlHttp.responseText);
                if (response){
                    loginSucces();
                    updateParameters();
                } else {
                    username = '';
                    alert('There has been an issue, likely that somone else is already using the username you have given, please try again with another username.');
                }
            } 
        }

        xmlHttp.open("POST", `/${username}/new_user`, true); // true for asynchronous 
        xmlHttp.send(null);
    } else {
        alert('Input empty please provide a username.')
    }
}

function login()
{
    let input = document.getElementById('usernameInput');
    let loginArea = document.getElementById('loginArea');
    username = input.value;

    if (username != '') {
        let xmlHttp = new XMLHttpRequest();
        xmlHttp.onreadystatechange = function() { 
            if (xmlHttp.readyState == 4 && xmlHttp.status == 200){
                let response = parseInt(xmlHttp.responseText); //1 is "good response" rather than saying the user exists
                if (response){
                    loginSucces();
                } else {
                    username = '';
                    alert('the ticket with this username is no longer open please open a new ticket to upload');
                }
            } 
        }

        xmlHttp.open("POST", `/${username}/login`, true); // true for asynchronous 
        xmlHttp.send(null);
    } else {
        alert('Input empty please provide a username.')
    }
}

function loginSucces(){
    alert(`logged in as ${username}!`);
    loggedIn = true;
    loginArea.style.display = 'none';
    document.getElementById('userLink').innerHTML = username;
    getParameters();
    uploadCheck();
}

function updateParameters(){
    //send maths parameters the user wants from the number inputs

    xLength = document.getElementById('xLength').value.toString();
    yLength = document.getElementById('yLength').value.toString();
    zLength = document.getElementById('zLength').value.toString();
    resolution = document.getElementById('resolution').value.toString();
    order = document.getElementById('order').value.toString();

    xNeg = document.getElementById('xNeg').value.toString();
    xPos = document.getElementById('xPos').value.toString();
    yNeg = document.getElementById('yNeg').value.toString();
    yPos = document.getElementById('yPos').value.toString();
    zNeg = document.getElementById('zNeg').value.toString();
    zPos = document.getElementById('zPos').value.toString();

    let paramsObj = {
        "xLength": xLength,
        "yLength": yLength,
        "zLength": zLength,
        "resolution": resolution,
        "order": order,

        "xNeg": xNeg,
        "xPos": xPos,
        "yNeg": yNeg,
        "yPos": yPos,
        "zNeg": zNeg,
        "zPos": zPos
    }

    let xmlHttp = new XMLHttpRequest();
    xmlHttp.onreadystatechange = function() { 
        if (xmlHttp.readyState == 4 && xmlHttp.status == 200){
            let dataStored = parseInt(xmlHttp.responseText);
            if (dataStored){
                console.log(`new parameters uploaded for ${username}`);
                alert('Parameters Updated!')
            } else {
                alert('problem when uploading parameters');
            }
        } 
    }

    xmlHttp.open("POST", `/${username}/update_params`, true); // true for asynchronous
    xmlHttp.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
    xmlHttp.send(JSON.stringify(paramsObj));
}

function getParameters(){
    //send to /set_defaults/username?foo=bar&bing=bong...
    let xmlHttp = new XMLHttpRequest();
    xmlHttp.onreadystatechange = function() { 
        if (xmlHttp.readyState == 4 && xmlHttp.status == 200){
            let paramsObj = JSON.parse(xmlHttp.responseText);
            
            xLength = paramsObj.xLength;
            yLength = paramsObj.yLength;
            zLength = paramsObj.zLength;
            resolution = paramsObj.resolution;
            order = paramsObj.order;

            xNeg = paramsObj.xNeg;
            xPos = paramsObj.xPos;
            yNeg = paramsObj.yNeg;
            yPos = paramsObj.yPos;
            zNeg = paramsObj.zNeg;
            zPos = paramsObj.zPos;

            document.getElementById('xLength').value = xLength;
            document.getElementById('yLength').value = yLength;
            document.getElementById('zLength').value = zLength;
            document.getElementById('resolution').value = resolution;
            document.getElementById('order').value = order;

            document.getElementById('xNeg').value = xNeg;
            document.getElementById('xPos').value = xPos;
            document.getElementById('yNeg').value = yNeg;
            document.getElementById('yPos').value = yPos;
            document.getElementById('zNeg').value = zNeg;
            document.getElementById('zPos').value = zPos;
        } 
    }

    xmlHttp.open("GET", `/${username}/get_params`, true); // true for asynchronous
    xmlHttp.send();
}

function upload(){
    //upload file from file input when upload button pressed

    let file = document.getElementById('uploadFileInput').files[0];

    let formData = new FormData();
    formData.append('file', file);

    let xmlHttp = new XMLHttpRequest();
    xmlHttp.onreadystatechange = function() { 
        if (xmlHttp.readyState == 4 && xmlHttp.status == 200){
            let uploadComplete = parseInt(xmlHttp.responseText);
            if (uploadComplete){
                alert('You are now in the queue! Every 20 seconds, while this browser is open, it will check to see if your file has been converted, you can leave now and log back in at a later point to check for your file.');
                uploadCheck();
            } else {
                alert('you already have a ticket waiting for processing');
            }
        } 
    }

    xmlHttp.open("POST", `/${username}/upload`, true); // true for asynchronous
    xmlHttp.send(formData);
}

function uploadCheck(){
    //check to see if user has a ticket waiting before constantly asking the server for the file

    let xmlHttp = new XMLHttpRequest();
    xmlHttp.onreadystatechange = function() { 
        if (xmlHttp.readyState == 4 && xmlHttp.status == 200){
            let response = parseInt(xmlHttp.responseText);
            if (response){
                document.getElementById('uploadFileInput').style.display = 'none';
                fileCheck();
            }
        } 
    }

    xmlHttp.open("GET", `/${username}/has_ticket`, true); // true for asynchronous
    xmlHttp.send();
}

function fileCheck(){
    //check to see if output file is done every 20s
    console.log('checking for output file every 20s');

    sendCheck();
    const fcInterval = setInterval(() => {
        if (!downloadReady){
            sendCheck();
        } else {
            clearInterval(fcInterval);
        }
    }, 20000);
}

function sendCheck(){
    let xmlHttp = new XMLHttpRequest();
    let fileReady = 0;

    xmlHttp.onreadystatechange = function() { 
        if (xmlHttp.readyState == 4 && xmlHttp.status == 200){
            fileReady = parseInt(xmlHttp.responseText);
            if (fileReady){
                downloadReady = true;

                alert('Your download is ready click on the download button to begin');

                let div = document.getElementById('downloadBtnDiv');
                let btn = document.createElement('button');

                btn.innerHTML = 'Download';
                btn.addEventListener('click', function(){
                    window.open(`/${username}/download`, '_blank');
                    this.remove();
                    document.getElementById('fileInputDiv').style.display = 'inline-block';
                });

                div.appendChild(btn);
            }
        }
    }

    console.log('asking server if output file is ready');

    xmlHttp.open("GET", `/${username}/file_check`, true); // true for asynchronous
    xmlHttp.send();
}

function newFrame(){
    //top canvas
    let n = 0;
    for (z = 0; z < zLeng_vis; z ++){
        for (x = 0; x < xLeng_vis; x++){
            topCtx.fillStyle = topFrames[currentFrame][n]; //get hsl string, points/frame * no frames to get to current start
            topCtx.fillRect(z * boxSize,((xLeng_vis-1) - x) * boxSize, boxSize, boxSize); //z is x axis, x is 1-x because canvas y axis is inverted
            n++;
        }
    }

    //breadth canvas
    n = 0;
    for (y = 0; y < yLeng_vis; y ++){
        for (x = 0; x < xLeng_vis; x++){
            breadthCtx.fillStyle = breadthFrames[currentFrame][n]; //get hsl string, points/frame * no frames to get to current start
            breadthCtx.fillRect(((yLeng_vis-1) - y) * boxSize,((xLeng_vis-1) - x) * boxSize, boxSize, boxSize); //z is x axis, x is 1-x because canvas y axis is inverted
            n++;
        }
    }

    //length canvas
    n = 0;
    for (z = 0; z < zLeng_vis; z ++){
        for (y = 0; y < yLeng_vis; y++){
            lengCtx.fillStyle = lengthFrames[currentFrame][n]; //get hsl string, points/frame * no frames to get to current start
            lengCtx.fillRect(z * boxSize,((yLeng_vis-1) - y) * boxSize, boxSize, boxSize); //z is x axis, x is 1-x because canvas y axis is inverted
            n++;
        }
    }
    
    currentFrame++;
    if (currentFrame >= frameCount){
        currentFrame = 0;
        clearInterval(intervalID);
    }
}

function getHSL(input) {

    let out = 120 * (input / 32767);
  
    // Return HSL string
    return `hsl(${out}, 100%, 50%)`;
}

function loadVisualiser(headerArr, visData){
    currentFrame = 0;

    //grabbing header data
    xLeng_vis = headerArr[0];
    yLeng_vis = headerArr[1];
    zLeng_vis = headerArr[2];
    let audioLeng = headerArr[3];
    frameLeng = headerArr[4];
    console.log(`x: ${xLeng_vis} y: ${yLeng_vis} z: ${zLeng_vis}, audioLeng: ${audioLeng} frameLeng: ${frameLeng}`);

    frameCount = Math.floor(audioLeng/frameLeng);
    // if (audioLeng % frameLeng != 0){
    //     frameCount += 1;
    // }

    let pointCount = xLeng_vis * yLeng_vis * zLeng_vis;
    let dataLeng = pointCount * frameCount;
    console.log(`frameCount: ${frameCount} pointCount: ${pointCount} dataLeng: ${dataLeng}`);

    let pointArr = [pointCount];

    let topArea = zLeng_vis * xLeng_vis;
    let breadthArea = yLeng_vis * xLeng_vis;
    let lengthArea = zLeng_vis * yLeng_vis;

    topFrames = [];
    breadthFrames = [];
    lengthFrames = [];

    //getting point values for each frame
    for (f = 0; f < frameCount; f++){
        for (p = 0; p < pointCount; p++){
            let index = visData[f + (p*frameCount)];
            pointArr[p] = index;
        }
        // indexArr[f] = pointArr;

        let topPoints = [topArea]
        let breadthPoints = [breadthArea];
        let lengthPoints = [lengthArea];

        //initialise point visualiser point arrays with 0s
        for (n = 0; n < topArea; n++){
            topPoints[n] = 0;
        }
        for (n = 0; n < breadthArea; n++){
            breadthPoints[n] = 0;
        }
        for (n = 0; n < lengthArea; n++){
            lengthPoints[n] = 0;
        }

        //combine points for 2D viewing
        //top
        for (i = 0; i < zLeng_vis * xLeng_vis; i++){
            let val = 0;
            for (y = 0; y < yLeng_vis; y++){
                val += Math.abs(Math.floor((pointArr[i + (y * xLeng_vis)]) / yLeng_vis));
            }
            topPoints[i] = getHSL(val);
            // topPoints[i] = val;
        }
        topFrames.push(topPoints);
        
        //breadth
        for (i = 0; i < yLeng_vis * xLeng_vis; i++){
            let val = 0;
            for (z = 0; z < zLeng_vis; z++){
                val += Math.abs(Math.floor((pointArr[i + (z * xLeng_vis * yLeng_vis)]) / zLeng_vis));
            }
            breadthPoints[i] = getHSL(val);
            // breadthPoints[i] = val;
        }
        breadthFrames.push(breadthPoints);
        
        //length
        for (i = 0; i < yLeng_vis * zLeng_vis; i++){
            let val = 0;
            for (x = 0; x < xLeng_vis; x++){
                val += Math.abs(Math.floor((pointArr[i + x]) / xLeng_vis));
            }
            lengthPoints[i] = getHSL(val);
            // lengthPoints[i] = val;
        }
        lengthFrames.push(lengthPoints);
    }


    let canvasTop = document.querySelector("#canvasTop");
    let canvasBreadth = document.querySelector("#canvasBreadth");
    let canvasLength = document.querySelector("#canvasLength");

    //calculate height and width values in percentages
    let canvasZLeng = ((zLeng_vis / (zLeng_vis+yLeng_vis)) * 100) - 10;    
    let canvasXLeng = ((xLeng_vis / (xLeng_vis+yLeng_vis)) * 100) - 10;
    let canvasYLeng;

    //fit longest side into canvas container
    if ((xLeng_vis + yLeng_vis) > (zLeng_vis + yLeng_vis)){
        canvasYLeng = ((yLeng_vis / (zLeng_vis+yLeng_vis)) * 100) - 10;
    } else {
        canvasYLeng = ((yLeng_vis / (xLeng_vis+yLeng_vis)) * 100) - 10;
    }

    canvasTop.style = `width: ${canvasZLeng}%; height: ${canvasXLeng}%;`;
    canvasBreadth.style = `width: ${canvasYLeng}%; height: ${canvasXLeng}%;`;
    canvasLength.style = `width: ${canvasZLeng}%; height: ${canvasYLeng}%;`;
    boxSize = canvasTop.width / zLeng_vis;

    //this is so fucking stupid but canvas needs to have direct values for scaling purposes
    canvasTop.width = boxSize * zLeng_vis;
    canvasTop.height = boxSize * xLeng_vis;
    canvasBreadth.width = boxSize * yLeng_vis;
    canvasBreadth.height = boxSize * xLeng_vis;
    canvasLength.width = boxSize * zLeng_vis;
    canvasLength.height = boxSize * yLeng_vis;

    // Initialize the GL context
    topCtx = canvasTop.getContext("2d");
    breadthCtx = canvasBreadth.getContext("2d");
    lengCtx = canvasLength.getContext("2d");

    // Only continue if WebGL is available and working
    if (topCtx === null) {
        alert("Unable to initialize WebGL. Your browser or machine may not support it.");
        return;
    }

    // Set clear color to grey, opaque
    topCtx.fillStyle = '#808080';
    topCtx.fillRect(0,0,canvasTop.width,canvasTop.height);
    breadthCtx.fillStyle = '#808080';
    breadthCtx.fillRect(0,0,canvasBreadth.width,canvasBreadth.height);
    lengCtx.fillStyle = '#808080';
    lengCtx.fillRect(0,0,canvasLength.width,canvasLength.height);

    // topCtx.clearColor(0.5, 0.5, 0.5, 1.0);
    // // Clear the color buffer with specified clear color
    // topCtx.clear(topCtx.COLOR_BUFFER_BIT);
    // breadthCtx.clearColor(0.5, 0.5, 0.5, 1.0);
    // breadthCtx.clear(breadthCtx.COLOR_BUFFER_BIT);
    // lengCtx.clearColor(0.5, 0.5, 0.5, 1.0);
    // lengCtx.clear(lengCtx.COLOR_BUFFER_BIT);
}

function newVisuFile(){
    visuFile = document.getElementById('playFileInput').files[0];
    let reader = new FileReader();

    reader.onload = function(){
        let data = reader.result;
        let intArr = new Uint8Array(data);

        for (i = 3; 0 < intArr.length; i++){
            if (intArr[i] == 117 && intArr[i - 1] == 115 && intArr[i - 2] == 105 && intArr[i - 3] == 118){
                
                let headerArr = intArr.slice(i + 1, i + 1 + headerSize);
                let visData = intArr.slice(i + 1 + headerSize + 4, intArr.length);

                headerArr = new Int32Array(headerArr.buffer);
                visData = new Int16Array(visData.buffer);

                loadVisualiser(headerArr, visData);
                break;
            }
        }
    };

    reader.readAsArrayBuffer(visuFile);
}

function playVisu(){
    let audio = new Audio(URL.createObjectURL(visuFile));

    newFrame();
    intervalID = setInterval(newFrame, frameLeng/44.1);

    audio.play();
}