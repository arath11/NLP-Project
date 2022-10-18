import { ViewChild } from '@angular/core';
import { Component, OnInit } from '@angular/core';
import { DrawableDirective } from './drawable.directive';

import * as tf from '@tensorflow/tfjs'

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})

export class AppComponent implements OnInit {
  //title = 'tensorflowApp';

  linearModel! : tf.Sequential;
  prediction:any;

  model: tf.LayersModel;
  predictions:any;

  //@ViewChild(DrawableDirective) canvas;//html canvas

  ngOnInit(){
    this.trainNewModel();
    this.loadModel();
  }
  
  //regrsion lineal 
  async trainNewModel(){
    this.linearModel=tf.sequential();
    this.linearModel.add(tf.layers.dense({units:1,inputShape:[1]}));

    //prepare the model
    this.linearModel.compile({loss:'meanSquaredError',optimizer:'sgd'});

    //entrenar
    const xs=tf.tensor1d([3.2,4.4,5.5,6.71,6.98,7.168,9.799,6.182,7.59,2.16]);
    const ys=tf.tensor1d([1.6,2.7,2.9,3.19,1.684,2.53,3.366,2.596,2.53,1.22 ]);

    await this.linearModel.fit(xs,ys);

    console.log('model trainged!!!!');    
        
    this.model= await tf.loadLayersModel('/assets/model.json');
    console.log('se importo el modelo');
  }
  
  linearPrediction(val:any){
    const output=this.linearModel.predict(tf.tensor2d([parseInt(val)],[1,1])) as any;
    this.prediction=Array.from(output.dataSync())[0]//para poder sacarlo y usarlo se requeire el array 
  }
  
  //cargar el modelo de los numeros 
  async loadModel(){
    this.model= await tf.loadLayersModel('/assets/model.json');
  }


  async predict(imageData:ImageData){
    //con tidy los tensores se borraran de la memoria ram cuando terminemos 
    const pred = await tf.tidy(() => {
      // Convert the canvas pixels to 
      let img = tf.browser.fromPixels(imageData, 1);
      img = img.reshape([1, 28, 28, 1]);//batch size, col, row, cantidad de colores (blanco y negro )
      img = tf.cast(img, 'float32'); //convertimos toda la info a float en los tensores

      // Make and format the predications
      const output = this.model.predict(img) as any;

      // Save predictions on the component
      this.predictions = Array.from(output.dataSync()); //regresa un tensor y lo convertimos a un arreglo 
      console.log(this.predictions)
    });

  }
  async predict1(data:any){
    //con tidy los tensores se borraran de la memoria ram cuando terminemos 
    console.log(data)
    
    const pred = await tf.tidy(() => {
      let dato = tf.tensor([[1,6,176,7,5063, 88, 12, 2679, 23,1310,  5, 109, 943,  4, 114,  9, 55 ,606,  5, 111,  7,4,139,193,273, 23, 4 ,172,
        , 270, 11, 7216,  4 ,8463, 2801,  109, 1603, 21,  4, 22 ,3861,  8,  6,
          1193, 1330,  4,  105,  987, 35,841, 19, 861, 1074,  5, 1987, 45, 55
        , 221, 15  ,670, 5304,  526, 14 ,1069,  4,  405,  5, 2438,  7, 27, 85
        , 108 , 131,  4, 5045 ,5304 ,3884 , 405,  9, 3523 , 133,  5, 50, 13,  104
        ,  51, 66 , 166, 14, 22  ,157,  9,  4,  530 , 239, 34 ,8463 ,2801, 45
        , 407, 31,  7, 41 ,3778 , 105, 21, 59 , 299, 12, 38  ,950,  5, 4521
        ,  15, 45  ,629  ,488, 2733  ,127,  6, 52  ,292, 17,  4, 6936 , 185 , 132,
          1988 ,5304 ,1799 , 488, 2693, 47,  6 , 392 , 173,  4 ,4378 , 270 ,2352,  4
          ,1500,  7,  4, 65, 55, 73, 11  ,346, 14, 20,  9,  6  ,976 ,2078
        ,   7 ,5293 , 861,  5 ,4182, 30 ,3127, 56,  4 , 841,  5  ,990,  692,  8
        ,   4 ,1669 , 398,  229, 13 ,2822 , 670 ,5304, 14,  9, 31,  7, 27 , 111
        , 108, 15 ,2033, 19, 7836, 1429 , 875 , 551, 14, 22,  9 ,1193, 21, 45
         , 4829,  5, 45  ,252,  8,  6 , 565 , 921 ,3639, 39,  4 , 529, 48, 25
        , 181,  8, 67, 35 ,1732, 22, 49 , 238, 60  ,135 ,1162, 14,  9 , 290
        ,   4, 58 , 472, 45, 55,  878,  8 , 169, 11  ,374 ,5687, 25  ,203, 28
        ,   8  ,818, 12 , 125,  4 ,3077,  0,  0,  0,  0,  0,  0,  0,  0
        ,   0,  0,  0,  0]])
      // Make and format the predications
      const output = this.model.predict(dato) as any;

      // Save predictions on the component
      this.predictions = Array.from(output.dataSync()); //regresa un tensor y lo convertimos a un arreglo 
      console.log(this.predictions)
    });

  }

}
