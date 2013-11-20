'use strict'

// Add string HashCode
String.prototype.hashCode = function(){
  var hash = 0;
  if (this.length == 0) return hash;
  for (var i = 0; i < this.length; i++) {
    var character = this.charCodeAt(i);
    hash = ((hash<<5)-hash)+character;
    hash = hash & hash; // Convert to 32bit integer
  }
  return hash;
};

var app = angular.module('annotateApp', [
  'annotateApp.directives',
  'annotateApp.controllers'
]);

app.run(function($rootScope) {
  $rootScope.$apply($(document).foundation());
});


var annotations = [
  { tag: "n", title: "Population Size" },
  { tag: "tx", title: "Shared intervention" },
  { tag: "tx1", title: "Intervention 1" },
  { tag: "tx2", title: "Intervention 2" },
  { tag: "tx3", title: "Intervention 3" },
  { tag: "tx4", title: "Intervention 4" },
  { tag: "tx5", title: "Intervention 5" },
  { tag: "n1", title: "Intervention 1 arm size" },
  { tag: "n2", title: "Intervention 2 arm size" },
  { tag: "n3", title: "Intervention 3 arm size" },
  { tag: "n4", title: "Intervention 4 arm size" },
  { tag: "n5", title: "Intervention 5 arm size" },
  { tag: "tx1-a", title: "Intervention 1 auxillary" },
  { tag: "tx2-a", title: "Intervention 2 auxillary" },
  { tag: "tx3-a", title: "Intervention 3 auxillary" },
  { tag: "tx4-a", title: "Intervention 4 auxillary" },
  { tag: "tx5-a", title: "Intervention 5 auxillary" }
];

app.constant("Annotations", annotations);
app.config(['$routeProvider', function($routeProvider) {
  $routeProvider.
    when('/visualize', {
      templateUrl: 'app/partials/visualize.html',
      controller: 'VisualizeController'
    }).
    when('/annotate', {
      templateUrl: 'app/partials/annotate.html',
      controller: 'AnnotateController'
    }).
    otherwise({
      redirectTo: '/visualize'
    });
}]);
