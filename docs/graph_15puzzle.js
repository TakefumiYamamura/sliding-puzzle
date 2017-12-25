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
 
    for (var i = 0; i < (lines.length - 1); i++) {
        // 1行ごとの処理
 
        var wordSet = lines[i].split(",");
 
        var wordData = {
            label: i,
            y: parseFloat(wordSet[0]),
        };
        // if (i == lines.length - 1) {break;}
 
        allData.push(wordData);
    }
    return allData;
}


window.onload = function () {
  var array1 = loadCSV("../result/korf100_result_speed_100.csv");
  // var array2 = loadCSV("../result/korf100_psimple_result.csv");
  var array3 = loadCSV("../result/korf100_psimple_result_50.csv");
  var array4 = loadCSV("../result/korf100_psimple_result_50_shared.csv");
  var array5 = loadCSV("../result/korf100_block_parallel_result_50.csv");
  var array6 = loadCSV("../result/korf100_block_parallel_result_with_staticlb_dfs_100_2048.csv");
  var array7 = loadCSV("../result/korf100_block_parallel_result_with_staticlb_100_2048.csv");
  var array8 = loadCSV("../result/korf100_result_expand_100.csv");
  var array8option = loadCSV("../result/korf100_result_expand_with_option100.csv");
  var cpuExpandOptionHorie = loadCSV("../result/idas_cpu_expand.txt");
  var array9 = loadCSV("../result/idas_smem.txt")
  var horieBestFindAll = loadCSV("../result/idas_bestall.txt")
  var trueBestFindAll = loadCSV("../result/korf100_block_parallel_result_with_staticlb_100_2048_all_true.csv")
  var array10 = loadCSV("../result/korf100_block_parallel_result_with_staticlb_100_2048_global.csv")
  var array11 = loadCSV("../result/korf100_block_parallel_result_with_staticlb_100_2048_all.csv")
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
    // {        
    //   type: "line",  
    //   name: "cpu",        
    //   showInLegend: true,
    //   dataPoints: array1
    // }, 
    // {        
    //   type: "spline",  
    //   name: "Psimple",        
    //   showInLegend: true,
    //   dataPoints: array2
    // },
    // {        
    //   type: "line",  
    //   name: "Psimple(constant)",        
    //   showInLegend: true,
    //   dataPoints: array3
    // },
    // {        
    //   type: "line",  
    //   name: "Psimple(shared)",        
    //   showInLegend: true,
    //   dataPoints: array4
    // },
    // {        
    //   type: "line",  
    //   name: "Block Parallel",        
    //   showInLegend: true,
    //   dataPoints: array5
    // },
    // {        
    //   type: "line",  
    //   name: "Block Parallel with staticlb dfs 2048",        
    //   showInLegend: true,
    //   dataPoints: array6
    // },
    // {        
    //   type: "line",  
    //   name: "Block Parallel with staticlb 2048 update finding one solution",        
    //   showInLegend: true,
    //   dataPoints: array7
    // },
    // {        
    //   type: "line",  
    //   name: "cpu expand wo option",        
    //   showInLegend: true,
    //   dataPoints: array8
    // },
    {        
      type: "line",  
      name: "cpu expand with option",        
      showInLegend: true,
      dataPoints: array8option
    },
    {        
      type: "line",  
      name: "cpu expand with option horie",        
      showInLegend: true,
      dataPoints: cpuExpandOptionHorie
    },
    // {        
    //   type: "line",  
    //   name: "Horie block parallel",        
    //   showInLegend: true,
    //   dataPoints: array9
    // },
    {        
      type: "line",  
      name: "Horie block parallel finding all best ver",        
      showInLegend: true,
      dataPoints: horieBestFindAll
    },
    {        
      type: "line",  
      name: "yamamura block parallel finding all best ver",        
      showInLegend: true,
      dataPoints: trueBestFindAll
    },
    // {        
    //   type: "line",  
    //   name: "Block Parallel with staticlb 2048 global",        
    //   showInLegend: true,
    //   dataPoints: array10
    // },
    // {        
    //   type: "line",  
    //   name: "Block Parallel with staticlb 2048 update all true",        
    //   showInLegend: true,
    //   dataPoints: array11
    // }
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
