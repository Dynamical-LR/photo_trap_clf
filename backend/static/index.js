window.addEventListener("DOMContentLoaded", (event) => {
	document.getElementById('upload-images').addEventListener("change", uploadImages);
	document.getElementById('select-images').addEventListener("click", selectImages);

	$(function() {
		//hang on event of form with id=myform
		$("#upload-images-form").submit(function(e) {
			e.preventDefault();    
			var formData = new FormData(this);

			$.ajax({
				url: window.location.pathname,
				type: 'POST',
				data: formData,
				success: function (data) {
					alert(data)
				},
				cache: false,
				contentType: false,
				processData: false
			});
			
				document.getElementById('upload-images-form').style.height = '50px';
				document.getElementById('result-window').style.display = 'block';
				document.getElementById('saved-time').style.display = 'block';
		});
	});
});


function selectImages() {
	document.getElementById("upload-images").click();
}

function uploadImages() {
	console.log('Uploading images')
	document.getElementById("submit-upload").click();
}
