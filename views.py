import datetime
import random

from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse
from django.shortcuts import render, redirect
from .models import *


media_path=r"D:\project\Risk_assessment\myapp\static\\"
# Create your views here.
def logg(request):
    return render(request, "login.html")


def login_post(request):
    username=request.POST['textfield']
    password=request.POST['textfield2']
    res=login.objects.filter(username=username, password=password)
    if res.exists():
        res=res[0]
        request.session['lid']=res.id
        if res.usertype=="admin":
            return HttpResponse("<script>alert('Welcome');window.location='/admin_home';</script>")
        elif res.usertype=="authority":
            return HttpResponse("<script>alert('Welcome');window.location='/auth_home';</script>")

        else:
            return HttpResponse("<script>alert('No access');window.location='/';</script>")

    else:
        return HttpResponse("<script>alert('Invalid details');window.location='/';</script>")


def admin_home(request):
    return render(request, "admin/index.html")


def adm_add_authority(request):
    return render(request, "admin/add_authority.html")

def adm_add_authority_post(request):
    name=request.POST['textfield']
    email=request.POST['textfield2']
    phone=request.POST['textfield3']
    gender=request.POST['radio']
    photo=request.FILES['fileField']

    psw=random.randint(1000, 9999)
    log_obj=login()
    log_obj.username=email
    log_obj.password=str(psw)
    log_obj.usertype="authority"
    log_obj.save()

    dt=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fs=FileSystemStorage()
    fs.save(media_path + "pic\\" + dt + ".jpg", photo)
    path="/static/pic/" + dt + ".jpg"
    obj=authority()
    obj.name=name
    obj.email=email
    obj.phone=phone
    obj.gender=gender
    obj.photo=path
    obj.LOGIN=log_obj
    obj.save()
    return HttpResponse("<script>alert('Authority added');window.location='/adm_add_authority';</script>")

def adm_view_authority(request):
    res=authority.objects.all()
    return render(request, "admin/view_authority.html", {'data' : res})

def adm_delete_authority(request, id):
    obj=authority.objects.get(id=id)
    obj.delete()
    return redirect("/adm_view_authority")

def adm_edit_authority(request, id):
    obj=authority.objects.get(id=id)
    return render(request, "admin/edit_authority.html", {'data' : obj})


def adm_edit_authority_post(request, id):
    name = request.POST['textfield']
    email = request.POST['textfield2']
    phone = request.POST['textfield3']
    gender = request.POST['radio']

    obj = authority.objects.get(id=id)
    obj.name = name
    obj.email = email
    obj.phone = phone
    obj.gender = gender

    if 'fileField' in request.FILES:
        photo = request.FILES['fileField']
        dt = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fs = FileSystemStorage()
        fs.save(media_path + "pic\\" + dt + ".jpg", photo)
        path = "/static/pic/" + dt + ".jpg"
        obj.photo = path
    obj.save()
    return HttpResponse("<script>alert('Authority updated');window.location='/adm_view_authority';</script>")


def adm_add_place(request):
    return render(request, "admin/add_place.html")

def adm_add_place_post(request):
    plc=request.POST['textfield']
    dist=request.POST['select']
    obj=places()
    obj.place_name=plc
    obj.district=dist
    obj.save()
    return HttpResponse("<script>alert('Place added');window.location='/adm_add_place';</script>")


def adm_view_place(request):
    res=places.objects.all()
    return render(request, "admin/view_place.html", {'data' : res})

def adm_delete_place(request, id):
    obj=places.objects.get(id=id)
    obj.delete()
    return redirect("/adm_view_place")

def adm_edit_place(request, id):
    obj=places.objects.get(id=id)
    return render(request, "admin/edit_place.html", {'data' : obj})

def adm_edit_place_post(request, id):
    plc = request.POST['textfield']
    dist = request.POST['select']

    obj = places.objects.get(id=id)
    obj.place_name = plc
    obj.district = dist
    obj.save()
    return HttpResponse("<script>alert('Place updated');window.location='/adm_view_place';</script>")

def adm_allocate_place(request):
    res=places.objects.all()
    res2=authority.objects.all()
    return render(request, "admin/allocate_place.html", {'data2':res, 'data': res2})

def adm_allocate_place_post(request):
    auth=request.POST['select']
    plc=request.POST['select2']
    res=allocation.objects.filter(AUTHORITY__id = auth, PLACE__id = plc)
    if res.exists():
        return HttpResponse("<script>alert('Already allocated');window.location='/adm_allocate_place';</script>")
    else:
        obj=allocation()
        obj.AUTHORITY_id = auth
        obj.PLACE_id = plc
        obj.save()
        return HttpResponse("<script>alert('Allocated successfully');window.location='/adm_allocate_place';</script>")

def adm_view_allocations(request):
    res=allocation.objects.all()
    return render(request, "admin/view_allocations.html", {'data': res})

def adm_delete_allocations(request, id):
    res=allocation.objects.get(id=id)
    res.delete()
    return redirect("/adm_view_allocations")

def adm_view_complaints(request):
    res=complaints.objects.all()
    return render(request, "admin/view_complaints.html", {'data' : res})

def adm_send_reply(request, id):
    return render(request, "admin/send_reply.html", {'id':id})
def adm_send_reply_post(request, id):
    rep=request.POST['textarea']
    obj=complaints.objects.get(id=id)
    obj.reply=rep
    obj.save()
    return HttpResponse("<script>alert('Reply sent');window.location='/adm_view_complaints';</script>")


def adm_view_suggestions(request):
    res=suggestions.objects.all()
    return render(request, "admin/view_suggestions.html", {'data' : res})

def adm_view_detections(request):
    res=detections.objects.all().order_by("-id")
    return render(request, "admin/view_detections.html", {"data":res})



def auth_home(request):
    return render(request, "authority/index.html")
    # return render(request, "authority/home.html")

def auth_view_allocation(request):
    res=allocation.objects.filter(AUTHORITY__LOGIN = request.session['lid'])
    return render(request, "authority/view_allocations.html", {'data' : res})

def auth_view_detections(request):
    res=allocation.objects.filter(AUTHORITY__LOGIN = request.session['lid'])
    res2=detections.objects.all().order_by("-id")
    ar=[]
    for i in res2:
        for j in res:
            if(i.CAMERA.PLACE==j.PLACE):
                ar.append(i)
    return render(request, "authority/view_detections.html", {"data":ar})


def auth_add_camera(request, id):
    return render(request, "authority/add_camera.html", {'id' : id})
def auth_add_camera_post(request, id):
    manu=request.POST['textfield']
    mod=request.POST['textfield2']
    area=request.POST['textfield3']
    all_obj=allocation.objects.get(id=id)
    obj=camera()
    obj.manufacturer=manu
    obj.cam_model=mod
    obj.area=area
    obj.PLACE = all_obj.PLACE
    obj.save()
    return HttpResponse("<script>alert('Camera added');window.location='/auth_view_allocation';</script>")

def auth_view_camera(request, id):
    all_obj=allocation.objects.get(id=id)
    res=camera.objects.filter(PLACE = all_obj.PLACE)
    return render(request, "authority/view_camera.html", {'data' : res, 'all_id' : id})

def auth_delete_camera(request, all_id, id):
    res=camera.objects.get(id=id)
    res.delete()
    return redirect("/auth_view_camera/"+all_id)

def auth_send_complaint(request):
    return render(request, "authority/send_complaint.html")
def auth_send_complaint_post(request):
    comp=request.POST['textarea']
    obj=complaints()
    obj.date= datetime.datetime.now().strftime("%Y-%m-%d")
    obj.complaint=comp
    obj.reply='pending'
    obj.AUTHORITY_id = request.session['lid']
    obj.save()
    return HttpResponse("<script>alert('Complaint sent');window.location='/auth_send_complaint';</script>")


def auth_view_reply(request):
    res=complaints.objects.filter(AUTHORITY__id=request.session['lid'])
    return render(request, "authority/view_reply.html", {'data':res})

def auth_delete_comp(request, id):
    obj=complaints.objects.get(id=id)
    obj.delete()
    return redirect("/auth_view_reply")

def auth_send_suggestion(request):
    return render(request, "authority/send_suggestion.html")
def auth_send_suggestion_post(request):
    sugg = request.POST['textarea']
    obj = suggestions()
    obj.date = datetime.datetime.now().date()
    obj.date = datetime.datetime.now().time()
    obj.suggestion = sugg
    obj.AUTHORITY_id = request.session['lid']
    obj.save()
    return HttpResponse("<script>alert('Suggestion sent');window.location='/auth_send_suggestion';</script>")


# def auth_view_detections(request):
#     res=allocation.objects.filter(AUTHORITY__LOGIN_id=request.session['lid'])
#     ar=[]
#     for j in res:
#         res2=detections.objects.filter(CAMERA__PLACE=j.PLACE)       #   view only detection in allocated places
#         for k in res2:
#             ar.append(k)
#     return render(request, "admin/view_detections.html", {"data":ar})
