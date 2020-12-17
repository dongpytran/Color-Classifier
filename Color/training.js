let data;
let model;
let xs, ys;
let rSlider, gSlider, bSlider;
let labelP;
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


function setup() {
  //Interface
  labelL = createP('Màu bạn chọn thuộc miền :');
  labelP = createP('---');
  labelP.style('font-size', '25px');
  labelR = createInput('R: ');
  labelR.style('width', '20px');
  labelR.style('border', 'none');
  labelR.style('font-size', '20px');
  labelR.style('color', '#000');
  labelR.style('background', 'transparent');
  labelR.attribute('disabled', '');
  rSlider = createSlider(0, 255, 255);
  rSlider.style('width', '150px');
  inputR2 = createInput();
  inputR2.id('rValue');
  inputR2.attribute('disabled', '');


  labelG = createInput('G: ');
  labelG.style('width', '20px');
  labelG.style('border', 'none');
  labelG.style('font-size', '20px');
  labelG.style('color', '#000');
  labelG.style('background', 'transparent');
  labelG.attribute('disabled', '');
  gSlider = createSlider(0, 255, 255);
  inputG2 = createInput();
  inputG2.id('gValue');
  inputG2.attribute('disabled', '');
  gSlider.style('width', '150px');


  labelB = createInput('B: ');
  labelB.style('width', '20px');
  labelB.style('border', 'none');
  labelB.style('font-size', '20px');
  labelB.style('color', '#000');
  labelB.style('background', 'transparent');
  labelB.attribute('disabled', '');
  bSlider = createSlider(0, 255, 255);
  inputB2 = createInput();
  inputB2.id('bValue');
  inputB2.attribute('disabled', '');
  bSlider.style('width', '150px');

  let colors = []; //khoi tao 1 mang luu tru cac mau
  let labels = []; // khoi tao 1 mang luu  tru nhan
  for (let record of data.entries) {//doc moi dong trong file data json
    let col = [record.r / 255, record.g / 255, record.b / 255]; //dua du lieu ve khoang 0-1
    colors.push(col);//add color vao mang colors
    labels.push(labelList.indexOf(record.label));//lay vi tri nhan cua mau do trong list nhan, add vao mang
  }
  console.log(colors);
  console.log(labels);
  xs = tf.tensor2d(colors); // convert mang colors sang kieu du lieu tensor
  let labelsTensor = tf.tensor1d(labels, 'int32'); //convert mang nhan sang kieu du lieu tensor
  console.log(xs.shape);
  ys = tf.oneHot(labelsTensor, 9).cast('float32');
  labelsTensor.dispose(); // loai bo tensor khoi bo nho
  console.log(ys.shape);
  //khoi tao model
  model = tf.sequential(); //mo hinh tuan tu: dau ra mot lop la dau vao cua lop tiep theo
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

  const LEARNING_RATE = 0.25; // Ty le hoc
  const optimizer = tf.train.sgd(LEARNING_RATE); // train theo rate
  //cau hinh model, cbi train
  model.compile({
    optimizer: optimizer, // trinh toi uu hoa
    loss: 'categoricalCrossentropy', //Giá trị mat mat sẽ được mô hình tối thiểu hóa,sau đó sẽ là tổng của tất cả các tổn thất riêng lẻ.
    metrics: ['accuracy'] //chỉ số được đánh giá bởi mô hình trong quá trình train va test
  });

  train();
  
}

async function train() {
  var percent = document.getElementById('percent');
  const options = {
    epochs : 50,
    validationSplit: 0.1, //data to validation: train 90%, kiem tra 10%;
    shuffle: true, // doi cho du lieu
    callbacks: {
      onTrainBegin: () => console.log('Training Start'), //bat dau train
      onTrainEnd: () => {yoBro()}, // ket thuc train
      onEpochEnd: (nums, logs) =>{ // voi moi lan training
        console.log('Epoch: '+ nums); // in ra lan train thu ?
        console.log('Loss:'+ logs.loss);// in ra mat mat
        var per = ((parseInt(nums)+2) * 2);
        percent.innerHTML = per + ' %';
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
    tf.tidy(() => { //Sử dụng phương pháp này giúp tránh rò rỉ bộ nhớ
      const input = tf.tensor2d([[r/255, g/255, b/255]]); //lay input tu silderbar
      let results = model.predict(input); //Thực hiện suy luận cho các tensor đầu vào.
      let argMax = results.argMax(1); //Trả về các chỉ số của giá trị lớn nhất (1) dọc theo nhan.
      let index = argMax.dataSync()[0];//đồng bộ các giá trị từ tf.Tensor
      //vd: 0 0 0 1 0 => index = 3
      let label = labelList[index];
      //lay label do
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
