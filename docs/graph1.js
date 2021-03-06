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
  var array1 = loadCSV("../result/korf100_burns_solver.csv");
  var array2 = loadCSV("../result/korf100_result8.csv");
  var array3 = loadCSV("../result/korf100_horie_solver.csv");
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
      type: "spline",  
      name: "Burns Solver",        
      showInLegend: true,
      dataPoints: array1
    }, 
    {        
      type: "spline",  
      name: "Yamamura Solver",        
      showInLegend: true,
      dataPoints: array2
    },
    {        
      type: "spline",  
      name: "spinute Solver",        
      showInLegend: true,
      dataPoints: array3
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
  chart.render();
}
