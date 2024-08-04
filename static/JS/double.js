document.addEventListener('DOMContentLoaded', function() {
    const dropdowns = document.querySelectorAll('.dropdown-wrapper');

    dropdowns.forEach(dropdown => {
        const toggleButton = dropdown.previousElementSibling;
        const menu = dropdown.querySelector('.dropdown-menu');

        toggleButton.addEventListener('click', function(event) {
            event.stopPropagation(); // Prevent click event from bubbling up
            menu.classList.toggle('hidden');
        });

        document.addEventListener('click', function(event) {
            if (!dropdown.contains(event.target)) {
                menu.classList.add('hidden');
            }
        });

        dropdown.addEventListener('mouseover', function() {
            menu.classList.remove('hidden');
        });

        dropdown.addEventListener('mouseleave', function() {
            menu.classList.add('hidden');
        });
    });
});

document.addEventListener('DOMContentLoaded', function () {
    const form = document.querySelector('#doubleDocumentForm');
    const submitFilesBtn = document.getElementById('submitDocsBtn');
    const compareDocsBtn = document.getElementById('compareDocsBtn');
    const inputDoc1 = document.getElementById('inputDoc1');
    const inputDoc2 = document.getElementById('inputDoc2');
    const displayDoc1 = document.getElementById('displayDoc1');
    const displayDoc2 = document.getElementById('displayDoc2');
    const resultsContainer = document.getElementById('resultsContainer');
    const fileInfo1 = document.getElementById('fileInfo1');
    const fileInfo2 = document.getElementById('fileInfo2');

    submitFilesBtn.disabled = true;
    compareDocsBtn.disabled = true;

    function checkFileInputs() {
        submitFilesBtn.disabled = !(inputDoc1.files.length > 0 && inputDoc2.files.length > 0);
    }

    function displayFileInfo(input, infoElement) {
        if (input.files.length > 0) {
            const file = input.files[0];
        } else {
            infoElement.textContent = '';
        }
    }

    inputDoc1.addEventListener('change', function () {
        displayFileInfo(inputDoc1, fileInfo1);
        checkFileInputs();
    });

    inputDoc2.addEventListener('change', function () {
        displayFileInfo(inputDoc2, fileInfo2);
        checkFileInputs();
    });

    submitFilesBtn.addEventListener('click', function (e) {
        e.preventDefault();
        loadFile(inputDoc1, displayDoc1);
        loadFile(inputDoc2, displayDoc2);
        compareDocsBtn.disabled = false;
    });

    function loadFile(input, display) {
        const file = input.files[0];
        if (file) {
            const reader = new FileReader();
            if (file.name.endsWith('.txt')) {
                reader.onload = function (e) {
                    display.textContent = e.target.result;
                };
                reader.readAsText(file);
            } else if (file.name.endsWith('.pdf')) {
                reader.onload = function (e) {
                    const pdfData = new Uint8Array(e.target.result);
                    pdfjsLib.getDocument({ data: pdfData }).promise.then(function (pdf) {
                        let textContent = "";
                        let pagePromises = [];
                        for (let pageNum = 1; pageNum <= pdf.numPages; pageNum++) {
                            pagePromises.push(pdf.getPage(pageNum).then(function (page) {
                                return page.getTextContent().then(function (text) {
                                    text.items.forEach(function (item) {
                                        textContent += item.str + " ";
                                    });
                                });
                            }));
                        }
                        Promise.all(pagePromises).then(function () {
                            display.textContent = textContent;
                        });
                    });
                };
                reader.readAsArrayBuffer(file);
            } else if (file.name.endsWith('.docx')) {
                reader.onload = function (e) {
                    mammoth.extractRawText({ arrayBuffer: e.target.result })
                        .then(function (result) {
                            display.textContent = result.value;
                        })
                        .catch(function (err) {
                            console.log(err);
                        });
                };
                reader.readAsArrayBuffer(file);
            }
        }
    }

    compareDocsBtn.addEventListener('click', function (e) {
        e.preventDefault();
        if (displayDoc1.textContent && displayDoc2.textContent) {
            submitForm();
        }
    });

    function submitForm() {
        const formData = new FormData(form);
        const file1 = inputDoc1.files[0];
        const file2 = inputDoc2.files[0];

        fetch('/doubleComparison', {
            method: 'POST',
            body: formData,
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                toastr.error(data.error, 'Error', { timeOut: 5000 });
            } else {
                displayResults(data.similarity, data.is_similar, data.original_text1, data.original_text2, data.highlighted_text1, data.highlighted_text2, file1, file2);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            toastr.error('An unexpected error occurred.', 'Error', { timeOut: 5000 });
        });
    }

    function displayResults(similarity, isSimilar, originalText1, originalText2, highlightedText1, highlightedText2, file1, file2) {
        resultsContainer.innerHTML = `
            <div class="text-lg p-4 mb-4 bg-blue-100 text-blue-800 rounded" role="alert">
                Similarity Score: ${similarity}%
            </div>
            <div class="text-lg p-4 mb-4 ${isSimilar ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'} rounded" role="alert">
                ${isSimilar ? 'The documents are similar.' : 'The documents are not similar.'}
            </div>

            <div class="container mt-5">
                <h1 class="text-3xl font-bold text-center text-blue-600">Text Comparison Highlighting</h1>
                <div class="flex flex-wrap mt-4">
                    <div class="w-full md:w-1/2 p-2">
                        <div class="bg-white shadow rounded overflow-y-auto h-72 p-4">
                            <h2 class="text-xl font-semibold mb-2">Highlighted Text 1</h2>
                            <div class="text-base p-2">${highlightedText1}</div>
                        </div>
                    </div>
                    <div class="w-full md:w-1/2 p-2">
                        <div class="bg-white shadow rounded overflow-y-auto h-72 p-4">
                            <h2 class="text-xl font-semibold mb-2">Highlighted Text 2</h2>
                            <div class="text-base p-2">${highlightedText2}</div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="container mt-4">
                <h1 class="text-3xl font-bold text-center text-blue-600">Documents Information</h1>
                <div class="flex justify-around items-center mt-4">
                    <div class="text-lg p-2">
                        <p><strong>File Name 1: ${file1.name}</strong></p>
                        <p><strong>File Size: ${(file1.size / 1024).toFixed(2)} KB</strong></p>
                    </div>
                    <div class="text-lg p-2">
                        <p><strong>File Name 2: ${file2.name}</strong></p>
                        <p><strong>File Size: ${(file2.size / 1024).toFixed(2)} KB</strong></p>
                    </div>
                </div>
            </div>
        `;

        resultsContainer.classList.remove('hidden');
        displayDoc1.textContent = originalText1;
        displayDoc2.textContent = originalText2;
    }
});
