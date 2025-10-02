const mongoose =require('mongoose')
const Schema=mongoose.Schema
const schema = mongoose.Schema(
    {
        name: String,
        nationalID:String, 
        address:String,
        phone:String,
        phone2:String,
        orders: [{type:Schema.Types.ObjectId,ref:"Order"}]
       }
);
module.exports=mongoose.model("Customer",schema);