function loadCSV(targetFile) {
 
    // 読み込んだデータを1行ずつ格納する配列
    var allData = [];
 
    // XMLHttpRequestの用意
    var request = new XMLHttpRequest();
    request.open("get", targetFile, false);
    request.send(null);
 
    // 読み込んだCSVデータ
    var csvData = request.responseText;
 
    // CSVの全行を取得
    var lines = csvData.split("\n");
 
    for (var i = 0; i < lines.length - 1; i++) {
        // 1行ごとの処理
 
        var wordSet = lines[i].split(",");
 
        var wordData = {
            label: i + 1,
            y: parseFloat(wordSet[0]),
        };
 
        allData.push(wordData);
    }
    return allData;
}

function divide_array(a, b) {
  var new_array = []
  for (var i = 0; i < a.length; i++) {
        var wordData = {
            label: i + 1,
            y: a[i]['y'] / b[i]['y'],
        };
        console.log(wordData);
 
        new_array.push(wordData);
    }
  return new_array;
}

window.onload = function () {
  var array1 = loadCSV("../result/yama24_med_result.csv");
  var array2 = loadCSV("../result/yama24_med_result_pdb_wo_cuda.csv");
  var array3 = loadCSV("../result/yama24_med_psimple_result.csv");
  var array4 = loadCSV("../result/yama24_med_psimple_with_pdb_result.csv");
  console.log(divide_array(array1, array2));
  var chart = new CanvasJS.Chart("chartContainer",
  {
    title:{
      text: "Result of IDA* in 24puzzle problems"             
    }, 
    animationEnabled: true,     
    axisY:{
      titleFontFamily: "arial",
      titleFontSize: 12,
      includeZero: false
    },
    toolTip: {
      shared: true
    },
    data: [
    {        
      type: "line",  
      name: "cpu",        
      showInLegend: true,
      dataPoints: array1
    }, 
    {        
      type: "line",  
      name: "cpu_pdb",        
      showInLegend: true,
      dataPoints: array2
    },
    {        
      type: "line",  
      name: "gpu",        
      showInLegend: true,
      dataPoints: array3
    }, 
    {        
      type: "line",  
      name: "gpu_pdb",        
      showInLegend: true,
      dataPoints: array4
    }
     
    ],
    legend:{
      cursor:"pointer",
      itemclick:function(e){
        if(typeof(e.dataSeries.visible) === "undefined" || e.dataSeries.visible) {
          e.dataSeries.visible = false;
        }
        else {
          e.dataSeries.visible = true;            
        }
        chart.render();
      }
    }
  });
  chart.render();
  var chart1 = new CanvasJS.Chart("chartContainer1",
  {
    title:{
      text: "cpu_PDB_speed_up in 24puzzle problems"             
    }, 
    animationEnabled: true,     
    axisY:{
      titleFontFamily: "arial",
      titleFontSize: 12,
      includeZero: false
    },
    toolTip: {
      shared: true
    },
    data: [
    {        
      type: "line",  
      name: "executed time ratio (cpu / cpu_pdb)",        
      showInLegend: true,
      dataPoints: divide_array(array1, array2)
    }
    ],
    legend:{
      cursor:"pointer",
      itemclick:function(e){
        if(typeof(e.dataSeries.visible) === "undefined" || e.dataSeries.visible) {
          e.dataSeries.visible = false;
        }
        else {
          e.dataSeries.visible = true;            
        }
        chart.render();
      }
    }
  });
  chart1.render();
  var chart2 = new CanvasJS.Chart("chartContainer2",
  {
    title:{
      text: "psimple_PDB_speed_up in 24puzzle problems"             
    }, 
    animationEnabled: true,     
    axisY:{
      titleFontFamily: "arial",
      titleFontSize: 12,
      includeZero: false
    },
    toolTip: {
      shared: true
    },
    data: [
    {        
      type: "line",  
      name: "executed time ratio (psimple gpu / psimple gpu pdb)",        
      showInLegend: true,
      dataPoints: divide_array(array3, array4)
    }
    ],
    legend:{
      cursor:"pointer",
      itemclick:function(e){
        if(typeof(e.dataSeries.visible) === "undefined" || e.dataSeries.visible) {
          e.dataSeries.visible = false;
        }
        else {
          e.dataSeries.visible = true;            
        }
        chart.render();
      }
    }
  });
  chart2.render();

  var chart3 = new CanvasJS.Chart("chartContainer3",
  {
    title:{
      text: "psimple_PDB_speed_up / cpu_PDB_speed_up in 24puzzle problems"             
    }, 
    animationEnabled: true,     
    axisY:{
      titleFontFamily: "arial",
      titleFontSize: 12,
      includeZero: false
    },
    toolTip: {
      shared: true
    },
    data: [
    {        
      type: "line",  
      name: "psimple_PDB_speed_up / cpu_PDB_speed_up",        
      showInLegend: true,
      dataPoints: divide_array(divide_array(array1, array2), divide_array(array3, array4))
    }, 
    ],
    legend:{
      cursor:"pointer",
      itemclick:function(e){
        if(typeof(e.dataSeries.visible) === "undefined" || e.dataSeries.visible) {
          e.dataSeries.visible = false;
        }
        else {
          e.dataSeries.visible = true;            
        }
        chart.render();
      }
    }
  });
  chart3.render();

  var chart4 = new CanvasJS.Chart("chartContainer4",
  {
    title:{
      text: "cpu_PDB_speed_up / psimple_PDB_speed_up in 24puzzle problems"             
    }, 
    animationEnabled: true,     
    axisY:{
      titleFontFamily: "arial",
      titleFontSize: 12,
      includeZero: false
    },
    toolTip: {
      shared: true
    },
    data: [
    {        
      type: "line",  
      name: "cpu_PDB_speed_up / psimple_PDB_speed_up",        
      showInLegend: true,
      dataPoints: divide_array(divide_array(array3, array4), divide_array(array1, array2))
    }, 
    ],
    legend:{
      cursor:"pointer",
      itemclick:function(e){
        if(typeof(e.dataSeries.visible) === "undefined" || e.dataSeries.visible) {
          e.dataSeries.visible = false;
        }
        else {
          e.dataSeries.visible = true;            
        }
        chart.render();
      }
    }
  });
  chart4.render();

  var chart5 = new CanvasJS.Chart("chartContainer-cpu",
  {
    title:{
      text: "Result of IDA*(without pdb) in 24puzzle problems"             
    }, 
    animationEnabled: true,     
    axisY:{
      titleFontFamily: "arial",
      titleFontSize: 12,
      includeZero: false
    },
    toolTip: {
      shared: true
    },
    data: [
    {        
      type: "line",  
      name: "cpu",        
      showInLegend: true,
      dataPoints: array1
    }, 
    {        
      type: "line",  
      name: "gpu",        
      showInLegend: true,
      dataPoints: array3
    }
    ],
    legend:{
      cursor:"pointer",
      itemclick:function(e){
        if(typeof(e.dataSeries.visible) === "undefined" || e.dataSeries.visible) {
          e.dataSeries.visible = false;
        }
        else {
          e.dataSeries.visible = true;            
        }
        chart.render();
      }
    }
  });
  chart5.render();

  var chart6 = new CanvasJS.Chart("chartContainer-gpu",
  {
    title:{
      text: "Result of IDA*(with pdb) in 24puzzle problems"             
    }, 
    animationEnabled: true,     
    axisY:{
      titleFontFamily: "arial",
      titleFontSize: 12,
      includeZero: false
    },
    toolTip: {
      shared: true
    },
    data: [
    {        
      type: "line",  
      name: "cpu_pdb",        
      showInLegend: true,
      dataPoints: array2
    }, 
    {        
      type: "line",  
      name: "gpu_psimple_pdb",        
      showInLegend: true,
      dataPoints: array4
    }
    ],
    legend:{
      cursor:"pointer",
      itemclick:function(e){
        if(typeof(e.dataSeries.visible) === "undefined" || e.dataSeries.visible) {
          e.dataSeries.visible = false;
        }
        else {
          e.dataSeries.visible = true;            
        }
        chart.render();
      }
    }
  });
  chart6.render();
}
