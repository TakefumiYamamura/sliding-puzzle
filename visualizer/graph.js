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
 
    for (var i = 0; i < lines.length; i++) {
        // 1行ごとの処理
 
        var wordSet = lines[i].split(",");
 
        var wordData = {
            x: i + 1,
            y: parseFloat(wordSet[0]),
        };
 
        allData.push(wordData);
    }
    return allData;
}
 


window.onload = function () {

  var array = loadCSV("../result/korf100_result.csv");
  console.log(array);
  var chart = new CanvasJS.Chart("chartContainer",{
    title: {
    text: "Result of IDA* in Kolf's 1000 problems"
    },
    axisX: {
    minimum: 1,
    maximum: 100
    },
    data: [
      {
        type: "spline",
        dataPoints: array
      }
    ]
  });
  chart.render();
  
  jQuery(".canvasjs-chart-canvas").last().on("click", 
    function(e){
      var parentOffset = $(this).parent().offset();
      var relX = e.pageX - parentOffset.left;
      var relY = e.pageY - parentOffset.top
      var xValue = Math.round(chart.axisX[0].convertPixelToValue(relX));
      var yValue = Math.round(chart.axisY[0].convertPixelToValue(relY));
    
      chart.data[0].addTo("dataPoints", {x: xValue, y: yValue});
      chart.axisX[0].set("maximum", Math.max(chart.axisX[0].maximum, xValue + 30));
    });
}