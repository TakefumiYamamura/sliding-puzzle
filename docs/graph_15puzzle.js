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


window.onload = function () {
  var array1 = loadCSV("../result/korf100_result_speed.csv");
  // var array2 = loadCSV("../result/korf100_psimple_result.csv");
  var array3 = loadCSV("../result/korf100_psimple_result_50.csv");
  var array4 = loadCSV("../result/korf100_psimple_result_50_shared.csv");
  var array5 = loadCSV("../result/korf100_block_parallel_result_50.csv");
  var chart = new CanvasJS.Chart("chartContainer",
  {
    title:{
      text: "Result of IDA* in Kolf's 100 problems"             
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
    // {        
    //   type: "spline",  
    //   name: "Psimple",        
    //   showInLegend: true,
    //   dataPoints: array2
    // },
    {        
      type: "line",  
      name: "Psimple(constant)",        
      showInLegend: true,
      dataPoints: array3
    },
    {        
      type: "line",  
      name: "Psimple(shared)",        
      showInLegend: true,
      dataPoints: array4
    },
    {        
      type: "line",  
      name: "Block Parallel",        
      showInLegend: true,
      dataPoints: array5
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
}
