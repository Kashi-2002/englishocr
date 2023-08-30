<style>
    h1{
        text-align: center;
        margin: 24px;
    }
    .group{
        text-align: center;
    margin-top: 25px;
    }
</style>

 
<h1>Convert handwritten pdf to typed text.</h1>
<form id="myForm" method="post" action="/newpage" enctype="multipart/form-data">
    <div class="group">
      <label for="file">Upload your file</label>
      <input
        type="file"
        id="file"
        name="fileToUpload"
      />
  
    <button type="submit" >Submit</button>
</div>
  </form>


  <script>






/**
   * @type {any}
   */
var res;



    async function extractor(){
      
// const form = document.getElementById("myForm");
// location.reload();
// console.log(form);
// @ts-ignore
// form.addEventListener("submit", function (event) {
//       event.preventDefault(); // Prevent the default form submission
//     });
    //   var mod=document.getElementById('file').value;
    //   var res = mod.split('\\')[mod.length()-1];
    //   console.log(res);
    //   fetch("http://127.0.0.1:8000/"+1 )
    // .then(async(response) => {
    //   let clone = response.clone();
    //   let res = await clone.json();
    //   console.log(res);
    //   console.log(clone)
    //   return response.blob()
    // })
    // .then(blob => {
    //   var url = window.URL.createObjectURL(blob);
    //   var a = document.createElement('a');
    //   a.href = url;
    //   a.download = "filename.xlsx";
    //   a.click();
    // })
    // .catch(function(err) {
    //   console.error(err)
    // })



    }


  </script>