/* -*- Mode: Java; tab-width: 2; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
/* vim: set shiftwidth=2 tabstop=2 autoindent cindent expandtab: */
/* Any copyright is dedicated to the Public Domain.
 * http://creativecommons.org/publicdomain/zero/1.0/ */
//

//
// bcw -- this is just slightly modified version of code (getinfo.js)
// distributed with pdfjs (in the examples directory).
// the assumption here is that the pdfjs directory lives in the same 
// directory and has the single file version has been built. In short
// do the following from the cochrane-nlp dir:
// 
// > git clone https://github.com/mozilla/pdf.js.git
// > cd pdfjs
// > node make singlefile
//
// usage: 
//      > node pdf_to_text.js "path/to/some.pdf" 
// this will dump text to standard out an dto a text file
// (path/to/some.pdf.txt). 
//
var fs = require('fs');
var atob = require('atob'); 

// HACK few hacks to let PDF.js be loaded not as a module in global space.
global.window = global;
global.navigator = { userAgent: "node" };
global.PDFJS = {};
global.DOMParser = require('./pdfjs/examples/node/domparsermock.js').DOMParserMock;

require('./pdfjs/build/singlefile/build/pdf.combined.js');

// Loading file from file system into typed array
var pdfPath = process.argv[2] || 'pdfjs/web/compressed.tracemonkey-pldi-09.pdf';
var outPath = pdfPath + ".txt";
var data = new Uint8Array(fs.readFileSync(pdfPath));

var entireDocStr = "";

// Will be using promises to load document, pages and misc data instead of
// callback.
PDFJS.getDocument(data).then(function (doc) {
  var numPages = doc.numPages;
  console.log('# Document Loaded');
  console.log('Number of Pages: ' + numPages);
  console.log();

  var lastPromise; // will be used to chain promises
  lastPromise = doc.getMetadata().then(function (data) {
    console.log('# Metadata Is Loaded');
    console.log('## Info');
    console.log(JSON.stringify(data.info, null, 2));
    console.log();
    if (data.metadata) {
      console.log('## Metadata');
      console.log(JSON.stringify(data.metadata.metadata, null, 2));
      console.log();
    }
  });

  var loadPage = function (pageNum) {
    return doc.getPage(pageNum).then(function (page) {
      //console.log('# Page ' + pageNum);
      var viewport = page.getViewport(1.0 /* scale */);
      //console.log('Size: ' + viewport.width + 'x' + viewport.height);
      //console.log();
      return page.getTextContent().then(function (content) {
        // Content contains lots of information about the text layout and
        // styles, but we need only strings at the moment
        var strings = content.items.map(function (item) {
          return item.str;
        });

        console.log('## Text Content');
        var pageText = strings.join(' ')
        console.log(pageText);
        entireDocStr += pageText + ' \n ';
      }).then(function () {
        console.log();
      });
    })
  };
  // Loading of the first page will wait on metadata and subsequent loadings
  // will wait on the previous pages.
  for (var i = 1; i <= numPages; i++) {
    lastPromise = lastPromise.then(loadPage.bind(null, i));
  }
  return lastPromise;
}).then(function () {
  //console.log('# End of Document');
  jsonified = JSON.stringify(entireDocStr);
  console.log(jsonified);
  fs.writeFile(outPath, jsonified);
  console.log('ok -- txt written to ' + outPath);
  //return pageText;
  //return JSON.stringify(entireDocStr);
}, function (err) {
  console.error('Error: ' + err);
});