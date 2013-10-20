'use strict';

/* Controllers */

var controllers = angular.module('annotateApp.controllers', []);


controllers.controller('AnnotateController', ['$scope', '$compile', 'Annotations', function($scope, $compile, Annotations) {
  $scope.fileName = "";
  $scope.file = {};
  $scope.text = "";
  $scope.result = "resultElement";

  $scope.upload = function(file) {
    $scope.text = file.contents;
    $scope.fileName = file.name.replace(/\.[^/.]+$/, "");
  };

  $scope.save = function(fileName, results) {
    var file = fileName + ".txt";
    var content = $("#" + results).html();
    var blob = new Blob([content], { type: "text/plain;charset=utf-8" });
    saveAs(blob, file);
  };

  /* From http://stackoverflow.com/questions/7380190/select-whole-word-with-getselection */
  var snapSelectionToWord = function() {
    var sel;

    // Check for existence of window.getSelection() and that it has a
    // modify() method. IE 9 has both selection APIs but no modify() method.
    if (window.getSelection && (sel = window.getSelection()).modify) {
      sel = window.getSelection();
      if (!sel.isCollapsed) {

        // Detect if selection is backwards
        var range = document.createRange();
        range.setStart(sel.anchorNode, sel.anchorOffset);
        range.setEnd(sel.focusNode, sel.focusOffset);
        var backwards = range.collapsed;
        range.detach();

        // modify() works on the focus of the selection
        var endNode = sel.focusNode, endOffset = sel.focusOffset;
        sel.collapse(sel.anchorNode, sel.anchorOffset);


        var direction = [];
        if (backwards) {
          direction = ['backward', 'forward'];
        } else {
          direction = ['forward', 'backward'];
        }

        sel.modify("move", direction[0], "character");
        sel.modify("move", direction[1], "word");
        sel.extend(endNode, endOffset);
        sel.modify("extend", direction[1], "character");
        sel.modify("extend", direction[0], "word");
      }
    }
    return sel;
  };

  var sel = {};
  var popup = $("#annotatePopup");

  $scope.annotate = function(annotation) {
    var range, node;
    if (typeof window.getSelection != "undefined") {

      // Test that the Selection object contains at least one Range
      if (sel.getRangeAt && sel.rangeCount) {
        // Get the first Range (only Firefox supports more than one)
        range = window.getSelection().getRangeAt(0);

        if (range.createContextualFragment) {
          var newNode = document.createElement(annotation);
          range.surroundContents(newNode);
          window.getSelection().removeAllRanges();
          popup.hide();
        }
      }
    };
  };

  $scope.annotations = Annotations;

  $scope.triggerAnnotate = function() {
    sel = snapSelectionToWord(); // should bubble to Controller closure
    if (sel.getRangeAt && sel.rangeCount) {
      var range = sel.getRangeAt(0);
      if(!_.isEmpty(sel.toString())) {
        var boundingBox = range.getClientRects();
        popup
          .css("top", boundingBox[0].top)
          .css("left", boundingBox[0].left)
          .css("display", "block");
      }
      else {
        popup.hide();
      }
    }
  };
}]);

controllers.controller('VisualizeController', ['$scope', function($scope) {
  $scope.files = [];

  $scope.addFile = function(files) {
    files.push({});
  };

  var splitAbstracts = _.memoize(function(str) {
    if(!str) return [];
    return str.split(/Abstract \d+ of \d+/);
  });

  /* wraps and processes an abstract string */
  var preProcess = function(str) {
    str = str.replace(/Notes: (.*)/, function(match, p1) {
      return '<br><br><span class="notes">' + match + "</span>";
    });
    return "<p>" + str + "</p>";
  };

  $scope.abstracts = _.memoize(function(files) {
    var results = [];
    for(var i = 0; i < files.length; i++) {
      var abstracts = splitAbstracts(files[i].contents);
      for(var j = 0; j < abstracts.length; j++) {
        if(!results[j]) {
          results[j] = {};
        }
        results[j][files[i].name] = preProcess(abstracts[j]);
      }
    }
    return results;
  }, function(arg) { // hash function
    return 31 * _.reduce(arg, function(memo, x) {
      return memo + (x.contents ? x.contents.hashCode() : -1);
    }, 1);
  });



}]);
