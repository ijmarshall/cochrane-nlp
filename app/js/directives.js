'use strict';

/* Directives */

var directives = angular.module('annotateApp.directives', []);

directives.directive('abstract', function ($compile) {
  return {
    scope: { content: "@",
             id: "@",
             editable: "@" },
    replace: true,
    restrict: 'E',
    template: "<div></div>",
    link: function(scope, element, attrs) {
      attrs.$observe('content', function(str) {
        if(str && str.length !== 0) {
          if(attrs.editable) {
            element.attr("contenteditable", true);
            element.attr("spellcheck", false);
          }
          element.attr("id", attrs.id);
          element.html(str);
          $compile(element.contents())(scope);
          if(!attrs.editable && str.indexOf("#EXCLUDE") != -1) {
            element.addClass("exclude");
          };
        }
      });
    }
  };
});

directives.directive('confirmationNeeded', function () {
  return {
    priority: 1,
    terminal: true,
    link: function (scope, element, attr) {
      var msg = attr.confirmationNeeded || "Are you sure?";
      var clickAction = attr.ngClick;
      element.bind('click',function () {
        if ( window.confirm(msg) ) {
          scope.$eval(clickAction);
        }
      });
    }
  };
});

directives.directive('fileReader', function () {
  return {
    scope: {
      file: '='
    },
    restrict: 'E',
    template: "<input type='file' onchange='angular.element(this).scope().upload(this)'>",
    link: function (scope, element, attrs) {
      scope.upload = function (element) {
        scope.$apply(function (scope) {
          scope.file = element.files[0];
        });
      };
      var filter = /^(text\/plain)$/i;

      scope.$watch('file', function (newVal, oldVal) {
        if(!newVal || !filter.test(newVal.type)) return;
        var reader = new FileReader();

        reader.onload = (function (file) {
          return function (env) {
            scope.$apply(function () {
              scope.file.contents = env.target.result;
            });
          };
        }(newVal));

        reader.readAsText(newVal);
      });
    }
  };
});

var createDirective = function(title) {
  return function() {
    return {
      scope: false,
      restrict: 'E',
      link: function(scope, element) {
        element.attr('data-tip', title);
      }
    };
  };
};

_.each(annotations, function(annotation) {
  directives.directive(annotation.tag, createDirective(annotation.title));
});
