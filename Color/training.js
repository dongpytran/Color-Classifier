let data;
let model;
let xs, ys;
let rSlider, gSlider, bSlider;
let labelP;
let lossP;
let labelL;
//list nhan mau
let labelList = [
  'red-ish',
  'green-ish',
  'blue-ish',
  'orange-ish',
  'yellow-ish',
  'pink-ish',
  'purple-ish',
  'brown-ish',
  'grey-ish'
];

//load data tu file json
function preload() {
  data = loadJSON('colorData.json');
}

//setup canvas, sliderbar
function setup() {
  // Crude interface
  labelL = createP('Màu bạn chọn thuộc miền :');
  labelP = createP('---');
  lossP = createP('loss');
  labelL = createP('R:');
  rSlider = createSlider(0, 255, 255);
  rSlider.style('width', '150px');
  inputR2 = createInput();
  inputR2.id('rValue');

  gSlider = createSlider(0, 255, 255);
  inputG2 = createInput();
  inputG2.id('gValue');
  gSlider.style('width', '150px');

  bSlider = createSlider(0, 255, 255);
  inputB2 = createInput();
  inputB2.id('bValue');
  bSlider.style('width', '150px');

  let colors = []; //khoi tao 1 mang luu tru cac mau
  let labels = []; // khoi tao 1 mang luu  tru nhan
  for (let record of data.entries) {//doc moi dong trong file data json
    let col = [record.r / 255, record.g / 255, record.b / 255]; //dua du lieu ve khoang 0-1
    colors.push(col);//add color vao mang colors
    labels.push(labelList.indexOf(record.label));//add nhan cua mau do(indexOf) vao list mau
  }

  xs = tf.tensor2d(colors);
  let labelsTensor = tf.tensor1d(labels, 'int32');

  ys = tf.oneHot(labelsTensor, 9).cast('float32');
  labelsTensor.dispose();

  //khoi tao model
  model = tf.sequential();
  //khoi tao lop hidden
  const hidden = tf.layers.dense({
    units: 16, //4, 16 , 32 tuy
    inputShape: [3], //input shape: du lieu dau vao: 3 thuoc tinh: r,g,b;
    activation: 'sigmoid' //ham kich hoat : sigmoid
  });
  //khoi tao lop output
  const output = tf.layers.dense({
    units: 9, //dau ra gom 9 labels tuong ung voi nhan dan trong file data
    activation: 'softmax' // ham kic hoat softmax
  });
  model.add(hidden); //add lop hidden vao model vua tao
  model.add(output); // add lop output vao model vua tao

  const LEARNING_RATE = 0.25; // toc do hoc
  const optimizer = tf.train.sgd(LEARNING_RATE); // train theo rate

  model.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });

  train();
  
}

async function train() {
  const options = {
    epochs : 50,
    validationSplit: 0.1, //data to validation: train 90%, kiem tra 10%;
    shuffle: true, // doi cho du lieu
    callbacks: {
      onTrainBegin: () => console.log('Training Start'), //bat dau train
      onTrainEnd: () => {yoBro()}, // ket thuc train
      onEpochEnd: (nums, logs) =>{ // voi moi lan training
        console.log('Epoch: '+ nums); // in ra lan train thu ?
        console.log('Loss:'+ logs.loss) // in ra mat mat
        lossP.html('Độ chính xác :' + (100-(logs.loss.toFixed(2))*10) + ' %');
      }
    }
    
  }
  return await model.fit(xs, ys, options); // bat dau train data
}

//sau khi train
function draw() {
  let r = rSlider.value();
  let g = gSlider.value();
  let b = bSlider.value();


  inputR2.value(r);
  inputG2.value(g);
  inputB2.value(b);
  if(r==255 && g==255 && b ==255){
    document.body.style.backgroundColor = '#fff';
  }else{
    document.body.style.backgroundColor = 'rgb(' + r + ',' + g + ',' + b + ')';
    tf.tidy(() => {
      const input = tf.tensor2d([[r/255, g/255, b/255]]); //lay input tu silderbar
      let results = model.predict(input); //du doan ket qua
      let argMax = results.argMax(1);
      let index = argMax.dataSync()[0];
      let label = labelList[index];
      labelP.html(label);
  });
  }
}

function yoBro(){
  var pre = document.getElementById("preloader2");
  console.log('Training Finished');
  alert('Data trained!')
  pre.style.display='none';
}
