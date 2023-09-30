window.addEventListener("DOMContentLoaded", (event) => {
	document.getElementById('upload-images').addEventListener("change", uploadImages);
	document.getElementById('select-images').addEventListener("click", selectImages);
});

function selectImages() {
	document.getElementById("upload-images").click();
}

function uploadImages() {
	console.log('Uploading images')
	document.getElementById('upload-images-form').style.height = '50px';
	document.getElementById('result-window').style.display = 'block';
	document.getElementById('saved-time').style.display = 'block';
}
