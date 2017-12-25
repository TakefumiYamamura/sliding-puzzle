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
            label: i,
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
            label: i,
            y: a[i]['y'] / b[i]['y'],
        };
        console.log(wordData);
 
        new_array.push(wordData);
    }
  return new_array;
}

window.onload = function () {
  var idas_cpu_horie = loadCSV("../result/idas_cpu_25.txt");

  var idas_cpu = loadCSV("../result/yama24_hard_new_expand_option_result.csv");
  var bpida = loadCSV("../result/yama24_hard_new_block_parallel_result_with_staticlb_dfs_100_2048.csv");
  var bpida_global = loadCSV("../result/yama24_hard_new_block_parallel_result_with_staticlb_dfs_100_2048_global.csv");


  var bpida_global_4  = loadCSV("../result/yama24_hard_new_block_parallel_result_with_staticlb_dfs_100_2048_global_152.csv");
  var bpida_global_9  = loadCSV("../result/yama24_hard_new_block_parallel_result_with_staticlb_dfs_100_2048_global_304.csv");
  var bpida_global_13 = loadCSV("../result/yama24_hard_new_block_parallel_result_with_staticlb_dfs_100_2048_global_456.csv");
  var bpida_global_18 = loadCSV("../result/yama24_hard_new_block_parallel_result_with_staticlb_dfs_100_2048_global_608.csv");
  var bpida_global_36 = loadCSV("../result/yama24_hard_new_block_parallel_result_with_staticlb_dfs_100_2048_global_1216.csv");


  var idas_cpu_pdb = loadCSV("../result/yama24_hard_new_result_pdb_expand.csv");
  var bpida_pdb = loadCSV("../result/yama24_hard_new_block_parallel_result_with_pdb_2048_dfs.csv");
  var bpida_global_pdb = loadCSV("../result/yama24_hard_new_block_parallel_result_with_pdb_2048_global.csv");
 

  var chart1 = new CanvasJS.Chart("result1",
  {
    title:{
      titleFontFamily: "verdana"
      // text: "Absolute time: IDA*(CPU) vs BPIDA*" 
      // text: ""             
    }, 
    animationEnabled: true,     
    axisY:{
      text: "Duration in seconds",
      titleFontFamily: "tahoma",
      titleFontSize: 6,
      includeZero: false
    },
    toolTip: {
      shared: true
    },
    data: [
    {        
      type: "line",  
      name: "IDA*(CPU)",        
      showInLegend: true,
      dataPoints: idas_cpu
    }, 
    {        
      type: "line",  
      name: "BPIDA*",        
      showInLegend: true,
      dataPoints: bpida
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
        chart1.render();
      }
    }
  });
  chart1.render();

  var chart2 = new CanvasJS.Chart("result2",
  {
    title:{
      titleFontFamily: "verdana"
      // text: "Absolute time: IDA*(CPU) vs BPIDA* vs BPIDA*GLOBAL" 
      // text: ""             
    }, 
    animationEnabled: true,     
    axisY:{
      text: "Duration in seconds",
      titleFontFamily: "tahoma",
      titleFontSize: 6,
      includeZero: false
    },
    toolTip: {
      shared: true
    },
    data: [
    {        
      type: "line",  
      name: "IDA*(CPU)",        
      showInLegend: true,
      dataPoints: idas_cpu
    }, 
    {        
      type: "line",  
      name: "BPIDA*",        
      showInLegend: true,
      dataPoints: bpida
    },
    {        
      type: "line",  
      name: "BPIDA* GLOBAL",        
      showInLegend: true,
      dataPoints: bpida_global
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
        chart2.render();
      }
    }
  });
  chart2.render();

  var chart3 = new CanvasJS.Chart("result3",
  {
    title:{
      titleFontFamily: "verdana"
      // text: "Absolute time: BPIDA*GLOBAL in different size of shared" 
      // text: ""             
    }, 
    animationEnabled: true,     
    axisY:{
      text: "Duration in seconds",
      titleFontFamily: "tahoma",
      titleFontSize: 6,
      includeZero: false
    },
    toolTip: {
      shared: true
    },
    data: [
    {        
      type: "line",  
      name: "BPIDA*GLOBAL 4Kbyte",        
      showInLegend: true,
      dataPoints: bpida_global_4
    }, 
        {        
      type: "line",  
      name: "BPIDA*GLOBAL 9Kbyte",        
      showInLegend: true,
      dataPoints: bpida_global_9
    }, 
    {        
      type: "line",  
      name: "BPIDA*GLOBAL 13Kbyte",        
      showInLegend: true,
      dataPoints: bpida_global_13
    }, 
    {        
      type: "line",  
      name: "BPIDA*GLOBAL 18Kbyte",        
      showInLegend: true,
      dataPoints: bpida_global_18
    },
    {        
      type: "line",  
      name: "BPIDA*GLOBAL 36Kbyte",        
      showInLegend: true,
      dataPoints: bpida_global_36
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
        chart3.render();
      }
    }
  });
  chart3.render();

  var chart4 = new CanvasJS.Chart("result4",
  {
    title:{
      titleFontFamily: "verdana"
      // text: "Absolute time: IDA*(CPU) vs BPIDA* vs BPIDA*GLOBAL in PDB" 
      // text: ""             
    }, 
    animationEnabled: true,     
    axisY:{
      text: "Duration in seconds",
      titleFontFamily: "tahoma",
      titleFontSize: 6,
      includeZero: false
    },
    toolTip: {
      shared: true
    },
    data: [
    {        
      type: "line",  
      name: "IDA*(CPU) pdb",        
      showInLegend: true,
      dataPoints: idas_cpu_pdb
    }, 
    {        
      type: "line",  
      name: "BPIDA* pdb",        
      showInLegend: true,
      dataPoints: bpida_pdb
    },
    {        
      type: "line",  
      name: "BPIDA* GLOBAL pdb",        
      showInLegend: true,
      dataPoints: bpida_global_pdb
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
        chart4.render();
      }
    }
  });
  chart4.render();



  var chart5 = new CanvasJS.Chart("cpuSpeedUp",
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
      dataPoints: divide_array(idas_cpu, idas_cpu_pdb)
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
        chart5.render();
      }
    }
  });
  chart5.render();

  var chart6 = new CanvasJS.Chart("bpidaSpeedUp",
  {
    title:{
      text: "BPIDA*_PDB_speed_up in 24puzzle problems"             
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
      name: "executed time ratio (BPIDA* manhattan / BPIDA* pdb)",        
      showInLegend: true,
      dataPoints: divide_array(bpida, bpida_pdb)
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
        chart6.render();
      }
    }
  });
  chart6.render();


  var chart7 = new CanvasJS.Chart("bpidaGlobalSpeedUp",
  {
    title:{
      text: "BPIDA*GLOBAL_PDB_speed_up in 24puzzle problems"             
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
      name: "executed time ratio (BPIDA*GLOBAL manhattan / BPIDA*GLOBAL pdb)",        
      showInLegend: true,
      dataPoints: divide_array(bpida_global, bpida_global_pdb)
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
        chart7.render();
      }
    }
  });
  chart7.render();


  var chart8 = new CanvasJS.Chart("speed_up_ratio_cpu_and_block_parallel",
  {
    title:{
      text: "cpu_speed_up / BPIDA*_PDB_speed_up in 24puzzle problems"             
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
      name: "executed time ratio (cpu_speed_up / BPIDA*_PDB_speed_up)",        
      showInLegend: true,
      dataPoints:  divide_array(divide_array(idas_cpu, idas_cpu_pdb), divide_array(bpida, bpida_pdb) )
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
        chart8.render();
      }
    }
  });
  chart8.render();

  var chart9 = new CanvasJS.Chart("speed_up_ratio_cpu_and_block_parallel_global",
  {
    title:{
      text: "cpu_speed_up / BPIDA*GLOBAL_PDB_speed_up in 24puzzle problems"             
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
      name: "executed time ratio (cpu_speed_up / BPIDA*GLOBAL_PDB_speed_up)",        
      showInLegend: true,
      dataPoints:  divide_array(divide_array(idas_cpu, idas_cpu_pdb), divide_array(bpida_global, bpida_global_pdb) )
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
        chart9.render();
      }
    }
  });
  chart9.render();

}
