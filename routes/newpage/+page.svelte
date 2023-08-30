<!-- <button type="submit" on:click={()=>extractor()}>Submit</button> -->


<script>
  import { onMount } from "svelte";




  const download = function (/** @type {BlobPart} */ data) {
  
  // Creating a Blob for having a csv file format 
  // and passing the data with type
  const blob = new Blob([data], { type: 'text/csv' });

  // Creating an object for downloading url
  const url = window.URL.createObjectURL(blob)

  // Creating an anchor(a) tag of HTML
  const a = document.createElement('a')

  // Passing the blob downloading url 
  a.setAttribute('href', url)

  // Setting the anchor tag attribute for downloading
  // and passing the download file name
  a.setAttribute('download', 'download.csv');

  // Performing a download with click
  a.click()
}



    onMount(async () => {
      const imageModules = import.meta.glob("/static/*.jpeg");
      // console.log(imageModules)

  for (const modulePath in imageModules) {
    imageModules[modulePath]().then(({ default: imageUrl }) => {
      var img=imageUrl.split("/")[2];
        fetch("http://127.0.0.1:8000/"+img )
    .then(async(response) => {
      let clone = response.clone();
      let res = await clone.json();
      // console.log(res);
      // const parsedData = parse(clone);
      var csv = res["hello"].map(function(d){
    return d.join();
}).join('\n');
      console.log(csv)
      download(csv);
      // const mainCSV = res.map(row => row.join(',')).join('\n');
      // console.log(clone)
      return response.blob()
    })
      // console.log({modulePath,img});
    });
  }
	});


  </script>