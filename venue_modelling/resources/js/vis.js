var visuFile;

var boxSize;
var currentFrame = 0;
var frameCount;
var frameLeng;
var intervalID;
var sampleRate;

var topFrames;
var breadthFrames;
var lengthFrames;

var topCtx;
var breadthCtx;
var lengCtx;

var headerSize = 20;

var playing = 0;

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
}

function newVisuFile(){
    visuFile = document.getElementById('playFileInput').files[0];
    let reader = new FileReader();

    pauseVisu();

    reader.onload = function(){
        let data = reader.result;
        let intArr = new Uint8Array(data);

        sampleRate = intArr.slice(24, 28);
        sampleRate = new Int32Array(sampleRate.buffer);
        sampleRate = sampleRate[0];

        //looking for visu tag
        let wavLeng = intArr.slice(40,44); //data size will always be @ pos 40 out of digest
        wavLeng = new Int32Array(wavLeng.buffer);
        wavLeng = wavLeng[0] + 48;

        let headerArr = intArr.slice(wavLeng, wavLeng + headerSize);
        let visData = intArr.slice(wavLeng + headerSize + 4, intArr.length);

        headerArr = new Int32Array(headerArr.buffer);
        visData = new Int16Array(visData.buffer);

        loadVisualiser(headerArr, visData);
    };

    reader.readAsArrayBuffer(visuFile);
}

var audio;
function playVisu(){
    if (!playing){
        audio = new Audio(URL.createObjectURL(visuFile));

        newFrame();
        intervalID = setInterval(newFrame, frameLeng/(sampleRate/1000));

        audio.play();

        playing = 1;
    }
}

function pauseVisu(){
    if (playing){
        audio.pause();
        currentFrame = 0;
        clearInterval(intervalID);
        playing = 0;
    }
}