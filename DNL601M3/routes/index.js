var express = require('express');
var router = express.Router();
const User = require('../model/user')
const Customer = require('../model/customer')
const Order = require('../model/order');
const multer  = require('multer')
const mongoose =require('mongoose') 
const path = require('path')
var fs = require('fs');

const storage = multer.diskStorage({
  destination: (req, file, cb) => {
      let destinationPath = path.join("public","images");
      if (!fs.existsSync(destinationPath)) {
          fs.mkdirSync(destinationPath);
      }
      cb(null, destinationPath);
  },
  filename: (req, file, cb) => {
    cb(null, file.originalname);
  }
});
const upload = multer({ storage }).single('file')


router.get('/newOrder',async function (req, res, next) {
  if(req.session.access===false){
    return res.render("error",{message:"لايمكنك إضافة طلب"})
  }
  let customers = await Customer.find()
  res.render('newOrder', { title: 'طلب جديد',customers:customers});
});

router.post('/newOrder',upload,async function (req, res, next) {
  var customer = await Customer.findOne({_id:req.body.customerId})
  var user = await User.findOne({_id:req.session.user_id})
  const order = new Order({
    file:req.file.originalname,
    description: req.body.description,
    status: req.body.status,
    notes: req.body.notes,
    insertD:new Date().toLocaleDateString("en-UK"),
    type: req.body.type,
    customer:customer,
    user:user
  });
  order.save();
  res.redirect("orders")
});





router.get('/', function (req, res, next) {
  if(req.session.user_id==undefined){
     return res.redirect("login")
  }
  res.render('index', { title: 'للعمل الإنساني AB' });
});






router.get('/regNew', function (req, res, next) {
  res.render('regNew', {layout:"layout2", title: 'تسجيل جديد' });
});
router.post('/regNew', function (req, res, next) {
  var access=true;
  if(req.body.access===undefined){
    access=false;
  }
  const newUser = new User({
        name: req.body.name,
        password: req.body.password,
        email: req.body.email,
        phone2: req.body.phone2,
        access: access,
        phone: req.body.phone
  });
  newUser.save();
  res.redirect("login");
});




router.get('/users', async (req, res) => {
  let users = await User.find()
  res.render('users', { title: 'المستخدمين',users:users});
})

router.get('/customers', async (req, res) => {
  let customer = await Customer.find()
  res.render('customers', { title: 'المراجعين',customer:customer});
})


router.get('/addCustomer', function (req, res, next) {
  res.render('addCustomer', { title: 'إضافة مراجع' });
});
router.post('/addCustomer', function (req, res, next) {
  const user = new Customer({
            name: req.body.name,
            nationalID: req.body.nationalID,
            phone: req.body.phone,
            phone2: req.body.phone2,
            address: req.body.address
  });
  user.save();
  res.render('addCustomer',{message:"تمت الإضافة بنجاح", title: 'إضافة مراجع'});
});






router.post('/search', async (req, res) => {
  let customer= await Customer.findOne({nationalID:req.body.keyword});
  let orders = await Order.find({customer:customer._id}).populate('customer').populate('user')
  res.render('search', { title: 'البحث',orders:orders});
})



router.get('/orders', async (req, res) => { 
  if(req.session.access==false){
    return res.render("error",{message:"لايمكنك الوصول إلى الطلبات"})
  }
    let orders = await Order.find().populate('customer').populate('user') 
    res.render('orders', { title: 'الطلبات',orders:orders});
})


router.get('/deleteOrder/:id', async (req, res) => {
    await Order.findByIdAndDelete(req.params.id);
    res.redirect("/orders")
})

router.get('/editOrder/:id', async (req, res) => {
  var order = await Order.findOne({_id:req.params.id})
  res.render('editOrder', { title: 'تعديل طلب',order:order });
})
router.post('/editOrder', async (req, res) => {
  var order = await Order.findOne({_id:req.params.id})
  var user = await User.findOne({_id:req.session.user_id})
  Order.updateOne({ _id: req.body.id }, {updateD:new Date().toLocaleString("en-UK"),
    status: req.body.status,description:req.body.description,notes:req.body.notes,type:req.body.type })  
    .then(result => {  
      res.redirect("/orders")
    })  
    .catch(err => {  
      res.render('editOrder', { title: 'تعديل طلب',order:order,message:"فشلت العملية" })
    }); 
})


router.get('/login', (req,res) =>{
  if( req.session.user_id!=undefined){
    res.redirect("/")
  } 
  res.render('login', { message:"تسجيل الدخول",title:"تسجيل الدخول",layout:'layout2'})
})

router.post('/login',async (req,res,next) =>{
  var user = await User.findOne({email: req.body.email,password:req.body.password})
  if(user==undefined){
      res.render('login', { title: "تسجيل الدخول",message:"فشلت  عملية التجيل، تأكد من المدخلات"})
    }
    else{
      req.session.access =user.access;
      req.session.user_id=user._id;
        res.redirect("/");
    }
})

router.get('/signup', (req,res) =>{
  req.session.access =undefined;
  req.session.user_id =undefined;
      res.redirect("login")
})



module.exports = router;
