let session;

const classNames = ['Fender Player Stratocaster', 'Fender Player Telecaster', 'Gibson Les Paul Standard 50s', 'Ibanez RG370DX', 'Ibanez S6570SK', 'Jackson Soloist SL2', 'PRS SE Custom 24'];
const modelOutputName = 'classifier_1';

async function initializeSession() {
  session = await ort.InferenceSession.create('http://kemmerling.me/guitarid_model_v0.0.1.onnx');
  console.log(session);
  document.getElementById('entry').innerHTML = "<h2>Guitar Identifier</h2><p>Upload an image of a guitar to identify its model and make</p><label for='file-input' class='btn-upload'>Click to Upload Guitar Image</label><input type='file' id='file-input' accept='image/*' onchange='handleImageUpload(event)'><div id='image-preview'></div><div class='output' id='output'></div>";
}

initializeSession();


function handleImageUpload(event) {
  const file = event.target.files[0];
  if (file) {
    const reader = new FileReader();
    reader.onload = function(e) {
      const img = new Image();
      img.src = e.target.result;
      console.log("target");
      console.log(e.target.result);
      console.log(img);
      document.getElementById('image-preview').innerHTML = '';
      document.getElementById('image-preview').appendChild(img);

      img.onload = function() {
        preprocessImage(img).then((inputTensor) => {
          runModel(inputTensor);
        });
      };
    };
    reader.readAsDataURL(file);
  }
}

async function preprocessImage(img) {
  // Remove alpha channel by drawing image to canvas and retrieving ImageData object instead of HTMLImage
  var canvas = document.createElement('canvas');
  var context = canvas.getContext('2d');

  canvas.width = img.width;
  canvas.height = img.height;
  context.drawImage(img, 0, 0, 224, 224);

  // document.body.appendChild(canvas);

  var myData = context.getImageData(0, 0, 224, 224);
  console.log("mydata");
  console.log(myData, img.width, img.height);



  // floatData = new Float32Array(myData);

  // for (let i = 0; i < floatData.length; i++) {
  //   floatData[i] = floatData[i] / 255.0;
  // }

  // console.log(floatData);

  // Create a canvas to draw the image
  const inputTensor = await ort.Tensor.fromImage(myData, options = {
    tensorFormat: "RGB",
    // norm: {
    // mean: [0.485, 0.456, 0.406],  // Normalization mean values for RGB
    // bias: [0.229, 0.224, 0.225],  // Normalization bias values (standard deviation)
    // },
  });

  // console.log(inputTensor);
  // const imageHTML = inputTensor.toImageData();
  // console.log(imageHTML);
  // // Display the image using a canvas
  // const canvas2 = document.createElement('canvas');
  // const context2 = canvas2.getContext('2d');
  // canvas2.width = imageHTML.width;
  // canvas2.height = imageHTML.height;
  // context2.putImageData(imageHTML, 0, 0);

  // Append the canvas to the DOM for visualization
  // document.body.appendChild(canvas2);
  console.log(inputTensor);
  return inputTensor;
}

function twoDecimals(num) {
  return Number(Math.round(num+'e2')+'e-2');
}

async function runModel(inputTensor) {
  try {
    const feeds = {'l_x_': inputTensor};
    const output = await session.run(feeds);
    const outputTensor = output[modelOutputName];

    if (!outputTensor) {
      document.getElementById('output').textContent = 'Output tensor is undefined or null.';
      return;
    }

    const arr = outputTensor.data;
    const index = arr.indexOf(Math.max(...arr));
    const class_name = classNames[index];

    document.getElementById('output').innerHTML = `Identified Guitar: <span class="prediction">${class_name}</span> <br /> Confidence: ${twoDecimals(arr[index])}`;
  } catch (error) {
    console.error('Error running the model:', error);
    document.getElementById('output').textContent = error;
  }
}