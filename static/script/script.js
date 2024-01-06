$("document").ready(function(){
    $("#send").click(function(){
    var grade= $('#Grade').val();
    var issue_d= $('#IssueDate').val();
    console.log(issue_d)
    var int_rate= $('#InterestRate').val();
    var annual_inc= $('#income').val();

    var data =  {
      "grade":grade,
      "issue_date":issue_d,
      "interest_rate":parseFloat(int_rate),
      "income":parseFloat(annual_inc)
  }
      $.ajax({
                  url: 'http://127.0.0.1:5000/predictdata',
                  type: "POST",
                  contentType: "application/json",
                  dataType: 'json',
                  data: JSON.stringify(data),
                  success: function(response) {
                    console.log(response)
                    var html= "<br><br><br><p> <b> RESULT : <b><p>";
                    response =response;
                         $.each(response,function(key,val){
                         console.log(val);
                            html+="<p>"+val+"<p>"
                        });
                        html +="<br>";
                        $(".show-data").append(html);
                      },
                  error: function(response){
                      console.log(response)
                  }
              })
    })
});